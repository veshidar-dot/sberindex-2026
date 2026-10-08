"""Детекторы на реальных данных: события из каталога и самые сильные тревоги.

    python -m sbi.detect_real

Порог каждого детектора - (1 - alpha)-квантиль максимальной оценки по случайной выборке рядов
(большинство рядов без сдвигов), т. е. тревога у доли alpha рядов за весь период.
Для рядов МО из каталога событий: была ли тревога в месяц события или в следующие 2, задержка,
и как меняется ответ BOCPD, если новость о событии известна (hazard_boost).
"""
import argparse

import numpy as np
import pandas as pd

from .config import data_path, load_config, out_path
from .data import CATEGORIES, load_panel
from .detect import DETECTORS
from .detect_eval import prepare
from .models.panel_ssm import fit
from .news import boost_matrix, load_events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    p = cfg["detection"]
    pr = p["real"]
    rng = np.random.default_rng(cfg["seed"])
    panel = load_panel(cfg)
    Z0, k, full = prepare(panel, pr["ref_months"])
    events = load_events(cfg)
    tid = panel.keys["territory_id"].to_numpy()
    ev_rows = np.where(full & np.isin(tid, np.concatenate(events.ids.to_numpy())))[0]
    sample = rng.choice(np.where(full)[0], pr["sample"], replace=False)
    rows = np.unique(np.r_[sample, ev_rows])
    Z = Z0[rows]
    cat = panel.cat[rows]
    news = boost_matrix(panel, events, p["news_factor"])[rows]
    start = pr["ref_months"]

    scores = {}
    for name in ("cusum", "pelt", "bocpd"):
        kw = dict(p["params"].get(name, {}))
        if "warm" in kw:
            kw["warm"] = start
        scores[name] = DETECTORS[name](Z, **kw)
    scores["bocpd_news"] = DETECTORS["bocpd"](Z, hazard_boost=news, **p["params"].get("bocpd", {}))
    S = np.zeros_like(Z)
    for c in range(6):
        m = cat == c
        lam, phi = fit(Z0[full & (panel.cat == c)])[:2]
        S[m] = DETECTORS["ssm_lr"](Z[m], lam, phi, w=p["window"], warm=start)
    scores["ssm_lr"] = S
    prior = np.zeros_like(Z)
    for t in range(Z.shape[1]):
        prior[:, t] = 2 * np.log(news[:, max(0, t - p["window"] + 1):t + 1].max(1))
    scores["ssm_lr_news"] = S + prior

    in_sample = np.isin(rows, sample)
    thr = {n: np.quantile(s[in_sample, start:].max(1), 1 - pr["alpha"]) for n, s in scores.items()}
    names = pd.read_csv(data_path(cfg, "mo_names")).set_index("territory_id").name

    out = []
    for _, ev in events.iterrows():
        t0 = panel.month_index(ev.event_month)
        if t0 < start:          # событие в опорном периоде: детектор ещё не работает
            continue
        for j in np.where(np.isin(tid[rows], ev.ids))[0]:
            r = {"event": ev.event_id, "territory_id": tid[rows[j]], "МО": names[tid[rows[j]]],
                 "category": CATEGORIES[cat[j]], "month": ev.event_month,
                 "shift_log": np.nanmean(panel.Y[rows[j], t0:t0 + 3]) - np.nanmean(panel.Y[rows[j], t0 - 3:t0])
                 - (np.nanmean(panel.Y[panel.cat == cat[j]][:, t0:t0 + 3]) - np.nanmean(panel.Y[panel.cat == cat[j]][:, t0 - 3:t0]))}
            for n, s in scores.items():
                hit = np.where(s[j, t0:t0 + p["window"]] > thr[n])[0]
                r[n] = int(hit[0]) if len(hit) else -1
            out.append(r)
    ev_tab = pd.DataFrame(out)
    ev_tab.to_csv(out_path(cfg, "detection_real_events.csv"), index=False, encoding="utf-8")
    pd.set_option("display.width", 250)
    print("Задержка тревоги в месяцах от месяца события (-1 = не обнаружено за 3 мес.):")
    print(ev_tab.round(3).to_string(index=False))
    print("\nДоля обнаруженных рядов событий:")
    print((ev_tab[list(scores)] >= 0).mean().round(3).to_string())

    # самые сильные тревоги в выборке (кандидаты для разбора)
    S = scores["ssm_lr"]
    top = []
    for j in np.argsort(-S[:, start:].max(1))[:pr["top"]]:
        t = start + int(S[j, start:].argmax())
        top.append({"territory_id": tid[rows[j]], "МО": names[tid[rows[j]]], "category": CATEGORIES[cat[j]],
                    "month": panel.months[t], "ssm_lr": S[j, t],
                    "jump_log": panel.Y[rows[j], t] - np.nanmean(panel.Y[rows[j], max(0, t - 3):t])})
    pd.DataFrame(top).to_csv(out_path(cfg, "detection_real_top.csv"), index=False, encoding="utf-8")
    pd.Series(thr).to_csv(out_path(cfg, "detection_real_thresholds.csv"), encoding="utf-8")


if __name__ == "__main__":
    main()
