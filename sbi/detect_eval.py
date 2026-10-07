"""Сравнение детекторов сдвигов на искусственно вставленных ступеньках.

    python -m sbi.detect_eval

1. Берём полные ряды (24 мес.), убираем общую компоненту категории n_t (среднее по МО в месяце t,
   считается только по месяцу t - без заглядывания вперёд).
2. Шум ряда k_i и опорный уровень - по первым ref_months месяцам; дальше ряд в единицах шума.
3. У доли inject_share рядов в случайный месяц tau вставляем ступеньку delta (в логах).
4. Каждый детектор даёт онлайн-оценку S[i, t]. Порог подбирается на рядах без вставки так,
   чтобы ложная тревога была у доли alpha рядов. При равных ложных тревогах сравниваем
   полноту, точность и задержку.
"""
import argparse
import time

import numpy as np
import pandas as pd

from .config import load_config, out_path
from .data import CATEGORIES, load_panel
from .detect import DETECTORS
from .models.panel_ssm import fit, series_scale


def prepare(panel, ref_months, rows=None):
    """Z в единицах шума ряда, центрированный по опорному периоду."""
    full = np.isfinite(panel.V).all(1)
    Y = panel.Y
    Z = np.full(Y.shape, np.nan)
    for c in range(6):
        m = full & (panel.cat == c)
        Z[m] = Y[m] - Y[m].mean(0)
    k = series_scale(Z[:, :ref_months])
    Z = (Z - np.nanmean(Z[:, :ref_months], 1, keepdims=True)) / np.sqrt(k)[:, None]
    return Z, k, full


def evaluate(S, inj, tau, start, alphas, w=3):
    """S (N, T) оценки; inj - есть ли вставка; tau - месяц вставки."""
    N, T = S.shape
    clean_max = S[~inj, start:].max(1)
    out = []
    for a in alphas:
        thr = np.quantile(clean_max, 1 - a)
        alarm = S > thr
        first = np.where(alarm[:, start:].any(1), alarm[:, start:].argmax(1) + start, -1)
        fp_clean = (first[~inj] >= 0).sum()
        ii = np.where(inj)[0]
        f = first[ii]
        early = (f >= 0) & (f < tau[ii])
        tp = (f >= tau[ii]) & (f < tau[ii] + w)
        delay = (f - tau[ii])[tp]
        out.append({"alpha": a, "threshold": thr, "recall": tp.mean(),
                    "precision": tp.sum() / max(tp.sum() + fp_clean + early.sum(), 1),
                    "delay": delay.mean() if len(delay) else np.nan,
                    "early_false": early.mean()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--detectors", nargs="+", default=list(DETECTORS))
    ap.add_argument("--per-category", type=int)
    a = ap.parse_args()
    cfg = load_config(a.config)
    p = cfg["detection"]
    if a.per_category:
        p["per_category"] = a.per_category
    rng = np.random.default_rng(cfg["seed"])
    panel = load_panel(cfg)
    Z0, k, full = prepare(panel, p["ref_months"])

    rows = []
    for c in range(6):
        cand = np.where(full & (panel.cat == c))[0]
        rows.append(rng.choice(cand, min(p["per_category"], len(cand)), replace=False))
    rows = np.concatenate(rows)
    Z = Z0[rows].copy()
    cat = panel.cat[rows]
    N, T = Z.shape
    inj = rng.random(N) < p["inject_share"]
    tau = rng.integers(p["tau_min"], p["tau_max"] + 1, N)
    delta = rng.choice(p["deltas"], N) * rng.choice([-1, 1], N)
    for i in np.where(inj)[0]:
        Z[i, tau[i]:] += delta[i] / np.sqrt(k[rows[i]])

    # новости: о доле news_true настоящих сдвигов известно в месяц tau, у доли news_false
    # рядов без сдвига - ложная новость в случайный месяц
    news = np.ones((N, T))
    has_news = inj & (rng.random(N) < p["news_true"])
    news[has_news, tau[has_news]] = p["news_factor"]
    fake = ~inj & (rng.random(N) < p["news_false"])
    news[fake, rng.integers(p["tau_min"], T, fake.sum())] = p["news_factor"]

    params = {}
    for c in range(6):
        params[c] = fit(Z0[full & (panel.cat == c)][:, :p["ref_months"] + 4])[:2]

    scores = {}
    for name in a.detectors:
        t0 = time.time()
        if name == "ssm_lr":
            S = np.zeros_like(Z)
            for c in range(6):
                m = cat == c
                S[m] = DETECTORS[name](Z[m], *params[c], w=p["window"], warm=p["tau_min"] - 2)
        else:
            S = DETECTORS[name](Z, **p["params"].get(name, {}))
        scores[name] = S
        print(f"{name}: {time.time() - t0:.0f} с", flush=True)
    # варианты с новостями
    if "bocpd" in scores:
        scores["bocpd_news"] = DETECTORS["bocpd"](Z, hazard_boost=news, **p["params"].get("bocpd", {}))
    if "ssm_lr" in scores:
        prior = np.zeros((N, T))
        for t in range(T):          # новость в последних window месяцах -> + 2 ln(factor) к LR
            prior[:, t] = 2 * np.log(news[:, max(0, t - p["window"] + 1):t + 1].max(1))
        scores["ssm_lr_news"] = scores["ssm_lr"] + prior

    res = []
    groups = {"все": np.ones(N, bool)}
    for d in sorted(set(np.abs(delta))):
        groups[f"|delta|={d:.2f}"] = (np.abs(delta) == d) | ~inj
    for c in range(6):
        groups[CATEGORIES[c]] = cat == c
    for name, S in scores.items():
        for g, m in groups.items():
            for r in evaluate(S[m], inj[m], tau[m], p["tau_min"], p["alphas"], p["window"]):
                res.append({"detector": name, "group": g, **r})
    res = pd.DataFrame(res)
    res.to_csv(out_path(cfg, "detection_synthetic.csv"), index=False, encoding="utf-8")
    np.savez_compressed(out_path(cfg, "detection_scores.npz"), rows=rows, inj=inj, tau=tau,
                        delta=delta, news=news, **scores)
    pd.set_option("display.width", 200)
    print(res[res.group == "все"].round(3).to_string(index=False))
    a5 = res[(res.alpha == 0.05) & res.group.str.startswith("|delta|")]
    print(a5.pivot(index="detector", columns="group", values="recall").round(3).to_string())


if __name__ == "__main__":
    main()
