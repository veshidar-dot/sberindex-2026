"""Сравнение детекторов сдвигов на искусственно вставленных ступеньках.

    python -m sbi.detect_eval               # все повторы из config (detection.seeds)
    python -m sbi.detect_eval --seeds 1     # один повтор

1. Берём полные ряды (24 мес.), убираем общую компоненту категории n_t (среднее по МО в месяце t,
   считается только по месяцу t - без заглядывания вперёд).
2. Шум ряда k_i и опорный уровень - по первым ref_months месяцам; дальше ряд в единицах шума.
3. У доли inject_share рядов в случайный месяц tau вставляем ступеньку delta (в логах).
4. Каждый детектор даёт онлайн-оценку S[i, t]. Порог подбирается на рядах без вставки так,
   чтобы ложная тревога была у доли alpha рядов. При равных ложных тревогах сравниваем
   полноту, точность и задержку.
5. Пространственные варианты (_spatial) работают с рядом «МО минус среднее k ближайших соседей
   по дорогам в той же категории» (соседи - из полной панели, их значения за тот же месяц):
   общие для местности колебания вычитаются, одиночный сдвиг МО виден лучше. _combo - тревога,
   если сработал обычный или пространственный вариант (максимум двух статистик LR).
6. Повторы с разными случайными вставками дают разброс; итог - среднее и 95 % интервал по повторам.
"""
import argparse
import time

import numpy as np
import pandas as pd
from scipy import stats

from .config import load_config, out_path
from .data import CATEGORIES, load_panel
from .detect import DETECTORS
from .models.panel_ssm import fit, series_scale


def _normalize(L, ref_months):
    k = series_scale(L[:, :ref_months])
    Z = (L - np.nanmean(L[:, :ref_months], 1, keepdims=True)) / np.sqrt(k)[:, None]
    return Z, k


def prepare(panel, ref_months, rows=None):
    """Z в единицах шума ряда, центрированный по опорному периоду."""
    full = np.isfinite(panel.V).all(1)
    Y = panel.Y
    L = np.full(Y.shape, np.nan)
    for c in range(6):
        m = full & (panel.cat == c)
        L[m] = Y[m] - Y[m].mean(0)
    Z, k = _normalize(L, ref_months)
    return Z, k, full


def prepare_relative(panel, cfg, ref_months, n_neighbors=10):
    """Ряд «МО минус среднее соседей по дорогам» в единицах собственного шума.
    Возвращает Zrel, krel (масштаб шума разности) и число соседей у каждого ряда."""
    from .graph import nearest_neighbors
    full = np.isfinite(panel.V).all(1)
    Y = panel.Y
    L = np.full(Y.shape, np.nan)
    for c in range(6):
        m = full & (panel.cat == c)
        L[m] = Y[m] - Y[m].mean(0)
    L = L - np.nanmean(L[:, :ref_months], 1, keepdims=True)
    tid = panel.keys["territory_id"].to_numpy()
    nn = nearest_neighbors(cfg, np.unique(tid), k=n_neighbors)
    pos = {(t, c): i for i, (t, c) in enumerate(zip(tid, panel.cat)) if full[i]}
    R = np.full(Y.shape, np.nan)
    cnt = np.zeros(panel.n, int)
    for i in np.where(full)[0]:
        js = [pos[(b, panel.cat[i])] for b in nn.get(tid[i], []) if (b, panel.cat[i]) in pos]
        if js:
            R[i] = L[js].mean(0)
            cnt[i] = len(js)
    Zrel, krel = _normalize(L - R, ref_months)
    return Zrel, krel, cnt


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


def _ssm_scores(Z, cat, params, p):
    S = np.zeros_like(Z)
    for c in range(6):
        m = cat == c
        S[m] = DETECTORS["ssm_lr"](Z[m], *params[c], w=p["window"], warm=p["tau_min"] - 2)
    return S


def run_once(cfg, panel, Z0, k, full, Zr0, kr, nbcnt, seed, detectors, verbose=True):
    p = cfg["detection"]
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(6):
        cand = np.where(full & (panel.cat == c) & (nbcnt > 0))[0]
        rows.append(rng.choice(cand, min(p["per_category"], len(cand)), replace=False))
    rows = np.concatenate(rows)
    Z, Zr = Z0[rows].copy(), Zr0[rows].copy()
    cat = panel.cat[rows]
    N, T = Z.shape
    inj = rng.random(N) < p["inject_share"]
    tau = rng.integers(p["tau_min"], p["tau_max"] + 1, N)
    delta = rng.choice(p["deltas"], N) * rng.choice([-1, 1], N)
    for i in np.where(inj)[0]:
        Z[i, tau[i]:] += delta[i] / np.sqrt(k[rows[i]])
        Zr[i, tau[i]:] += delta[i] / np.sqrt(kr[rows[i]])     # соседи не сдвигались

    # новости: о доле news_true настоящих сдвигов известно в месяц tau, у доли news_false
    # рядов без сдвига - ложная новость в случайный месяц
    news = np.ones((N, T))
    has_news = inj & (rng.random(N) < p["news_true"])
    news[has_news, tau[has_news]] = p["news_factor"]
    fake = ~inj & (rng.random(N) < p["news_false"])
    news[fake, rng.integers(p["tau_min"], T, fake.sum())] = p["news_factor"]
    prior = np.zeros((N, T))
    for t in range(T):          # новость в последних window месяцах -> + 2 ln(factor) к LR
        prior[:, t] = 2 * np.log(news[:, max(0, t - p["window"] + 1):t + 1].max(1))

    ref = p["ref_months"] + 4
    params = {c: fit(Z0[full & (panel.cat == c)][:, :ref])[:2] for c in range(6)}
    params_r = {c: fit(Zr0[full & (panel.cat == c) & (nbcnt > 0)][:, :ref])[:2] for c in range(6)}

    scores = {}
    for name in detectors:
        t0 = time.time()
        if name == "ssm_lr":
            scores[name] = _ssm_scores(Z, cat, params, p)
        else:
            scores[name] = DETECTORS[name](Z, **p["params"].get(name, {}))
        if verbose:
            print(f"  seed {seed} {name}: {time.time() - t0:.0f} с", flush=True)
    if "bocpd" in scores:
        scores["bocpd_news"] = DETECTORS["bocpd"](Z, hazard_boost=news, **p["params"].get("bocpd", {}))
        scores["bocpd_spatial"] = DETECTORS["bocpd"](Zr, **p["params"].get("bocpd", {}))
    if "ssm_lr" in scores:
        scores["ssm_lr_news"] = scores["ssm_lr"] + prior
        scores["ssm_lr_spatial"] = _ssm_scores(Zr, cat, params_r, p)
        scores["ssm_lr_spatial_news"] = scores["ssm_lr_spatial"] + prior
        scores["ssm_lr_combo"] = np.maximum(scores["ssm_lr"], scores["ssm_lr_spatial"])
        scores["ssm_lr_combo_news"] = scores["ssm_lr_combo"] + prior

    res = []
    groups = {"все": np.ones(N, bool)}
    for d in sorted(set(np.abs(delta))):
        groups[f"|delta|={d:.2f}"] = (np.abs(delta) == d) | ~inj
    for c in range(6):
        groups[CATEGORIES[c]] = cat == c
    for name, S in scores.items():
        for g, m in groups.items():
            for r in evaluate(S[m], inj[m], tau[m], p["tau_min"], p["alphas"], p["window"]):
                res.append({"seed": seed, "detector": name, "group": g, **r})
    return pd.DataFrame(res), dict(rows=rows, inj=inj, tau=tau, delta=delta, news=news, **scores)


def summarize(res, alpha=0.05, max_alpha=0.2):
    """Среднее и 95 % интервал по повторам; pAUC - средняя полнота на сетке долей ложных тревог <= max_alpha
    (площадь под кривой полнота-доля ложных тревог, делённая на max_alpha)."""
    rows = []
    for (det, grp), g in res.groupby(["detector", "group"]):
        per_seed = []
        for s, gs in g.groupby("seed"):
            gs = gs.sort_values("alpha")
            gs = gs[gs.alpha <= max_alpha]
            x, y = np.r_[0.0, gs.alpha], np.r_[0.0, gs.recall]
            pauc = np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2) / max_alpha
            at = gs[np.isclose(gs.alpha, alpha)].iloc[0]
            per_seed.append({"recall": at.recall, "precision": at.precision, "delay": at.delay, "pauc": pauc})
        d = pd.DataFrame(per_seed)
        n = len(d)
        q = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
        row = {"detector": det, "group": grp, "seeds": n}
        for col in d.columns:
            row[col] = d[col].mean()
            row[col + "_ci"] = q * d[col].std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--detectors", nargs="+", default=list(DETECTORS))
    ap.add_argument("--per-category", type=int)
    ap.add_argument("--seeds", type=int)
    a = ap.parse_args()
    cfg = load_config(a.config)
    p = cfg["detection"]
    if a.per_category:
        p["per_category"] = a.per_category
    n_seeds = a.seeds or p.get("seeds", 1)
    panel = load_panel(cfg)
    Z0, k, full = prepare(panel, p["ref_months"])
    Zr0, kr, nbcnt = prepare_relative(panel, cfg, p["ref_months"], p.get("neighbors", 10))
    print(f"шум ряда (медиана sd, логи): сам ряд {np.median(np.sqrt(k[full])):.3f}, "
          f"ряд минус соседи {np.median(np.sqrt(kr[full & (nbcnt > 0)])):.3f}", flush=True)

    all_res, first = [], None
    for s in range(n_seeds):
        res, sc = run_once(cfg, panel, Z0, k, full, Zr0, kr, nbcnt, cfg["seed"] + s, a.detectors)
        all_res.append(res)
        first = first or sc
    res = pd.concat(all_res, ignore_index=True)
    res.to_csv(out_path(cfg, "detection_synthetic.csv"), index=False, encoding="utf-8")
    np.savez_compressed(out_path(cfg, "detection_scores.npz"), **first)
    summ = summarize(res)
    summ.to_csv(out_path(cfg, "detection_summary.csv"), index=False, encoding="utf-8")
    pd.set_option("display.width", 220)
    cols = ["detector", "recall", "recall_ci", "precision", "delay", "pauc", "pauc_ci"]
    print(summ[summ.group == "все"][cols].sort_values("recall", ascending=False).round(3).to_string(index=False))
    print(summ[summ.group.str.startswith("|delta|")].pivot(index="detector", columns="group", values="recall").round(3).to_string())


if __name__ == "__main__":
    main()
