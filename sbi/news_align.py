"""Согласование новостей с данными СберИндекса: заголовки федеральных лент и постановления о режиме ЧС.

    python -m sbi.news_align

Источники (все с датой публикации; модель видит новость с конца месяца, в котором она вышла):
  * заголовки Интерфакса и Ленты 2023-2024 гг. с привязкой к МО (sbi.news_crawl, sbi.news_geo) -
    ref/news/headlines_monthly_mo.csv;
  * постановления о введении режима ЧС (a-amik/sberindex-2026, MIT) - ref/news/chs_decrees_amik.json.
Проверки:
  1. покрытие: сколько МО и МО-месяцев вообще попадают в новости, по типам событий;
  2. реакция трат: средний сдвиг ряда к своей категории (3 мес. после минус 3 мес. до, %) в МО-месяцы с новостью
     против тех же месяцев у МО без новости, 95 % интервал - бутстреп по МО;
  3. опережение: подъём вероятности подтверждённого сдвига рядом с новостью за 2 и 1 месяц, в тот же месяц и позже
     (как для GDELT: ожидаемая частота по месяцам, sbi.gdelt.event_study);
  4. детектор: LR-тест с новостью как априорной вероятностью сдвига против LR-теста без новостей при одинаковой
     доле месяцев с тревогой (sbi.gdelt.detector_with_hits);
  5. объяснимость: у какой доли тревог есть местная новость в месяц тревоги или за 2 месяца до неё, против той же
     доли в случайные месяцы тех же рядов.
"""
import json

import numpy as np
import pandas as pd

from .config import load_config, out_path, resolve
from .data import CATEGORIES, load_panel
from .detect_eval import prepare
from .gdelt import confirmed_shifts, detector_with_hits, event_study

GROUPS = {
    "все местные заголовки": None,
    "ЧС, паводок, пожар": ["disaster", "flood", "fire", "emergency"],
    "атаки БПЛА": ["attack"],
    "предприятия, увольнения": ["plant", "layoff"],
    "выплаты, компенсации": ["payments"],
    "транспорт": ["transport"],
}


def headline_matrices(panel):
    """Число местных заголовков группы в МО ряда за месяц: {группа: (ряды x месяцы)}."""
    h = pd.read_csv(resolve("ref/news/headlines_monthly_mo.csv"))
    h = h[h.month.isin(panel.months)]
    tid = panel.keys["territory_id"].to_numpy()
    out = {}
    for g, types in GROUPS.items():
        v = h["all"] if types is None else h[[t for t in types if t in h.columns]].sum(1)
        p = pd.DataFrame({"territory_id": h.territory_id, "month": h.month, "v": v})
        p = p.pivot_table(index="territory_id", columns="month", values="v", aggfunc="sum").reindex(columns=panel.months)
        M = p.reindex(tid).fillna(0.0).to_numpy()
        out[g] = M
    return out


def decree_matrix(panel):
    d = json.loads(resolve("ref/news/chs_decrees_amik.json").read_text(encoding="utf-8"))
    tid = panel.keys["territory_id"].to_numpy()
    M = np.zeros((panel.n, panel.T))
    for e in d:
        m = e["date"][:7]
        if m in panel.months:
            M[np.isin(tid, e["territory_ids"]), panel.months.index(m)] += 1
    return M


def response(panel, full, hit, boot, rng, within=False):
    """Средний сдвиг ряда к своей категории (%, 3 мес. после минус 3 до) в месяцы с новостью минус контроль.
    within=False: контроль - ряды без новости в те же месяцы (календарь); within=True: контроль - тот же ряд в месяцы
    без новости (убирает различие между крупными городами, о которых пишут, и малыми МО). Бутстреп по МО."""
    L = panel.Y.copy()
    for c in range(6):
        m = panel.cat == c
        L[m] -= np.nanmean(L[m & full], 0)
    T = panel.T
    sh = np.full(L.shape, np.nan)
    for t in range(3, T - 2):
        sh[:, t] = np.nanmean(L[:, t:t + 3], 1) - np.nanmean(L[:, t - 3:t], 1)
    ok = full[:, None] & np.isfinite(sh)
    treat = ok & (hit > 0)
    if treat.sum() < 10:
        return None
    if within:
        base = np.where(ok & ~treat, sh, np.nan)
        diff = sh - np.nanmean(base, 1, keepdims=True)
        ctrl = base - np.nanmean(base, 1, keepdims=True)
    else:
        ctrl_mean = np.array([np.nanmean(sh[ok[:, t] & ~treat[:, t], t]) if (ok[:, t] & ~treat[:, t]).any() else np.nan
                              for t in range(T)])
        diff = sh - ctrl_mean[None, :]
        ctrl = np.where(ok & ~treat, diff, np.nan)
    tid = panel.keys["territory_id"].to_numpy()
    ug, inv = np.unique(tid, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(ug))]
    def stat(rows):
        d = diff[rows][treat[rows]]
        return np.nanmean(d), np.nanmean(np.abs(d))
    est, est_abs = stat(np.arange(panel.n))
    bs = [stat(np.concatenate([members[g] for g in rng.integers(0, len(ug), len(ug))])) for _ in range(boot)]
    bs = np.array(bs)
    lo, hi = np.nanpercentile(bs[:, 0], [2.5, 97.5])
    alo, ahi = np.nanpercentile(bs[:, 1], [2.5, 97.5])
    rows_t = np.unique(np.where(treat)[0])
    ctrl_abs = np.nanmean(np.abs(ctrl[rows_t]))                # тот же модуль у тех же рядов в месяцы без новости
    return {"control": "тот же ряд без новости" if within else "другие ряды в тот же месяц","news_series_months": int(treat.sum()), "mo": int(len(np.unique(tid[np.where(treat)[0]]))),
            "shift_pct": 100 * est, "ci_low": 100 * lo, "ci_high": 100 * hi,
            "abs_shift_pct": 100 * est_abs, "abs_ci_low": 100 * alo, "abs_ci_high": 100 * ahi,
            "abs_shift_control_pct": 100 * ctrl_abs}


def main():
    cfg = load_config()
    pg = cfg["gdelt"]
    rng = np.random.default_rng(cfg["seed"])
    panel = load_panel(cfg)
    Z, _, full = prepare(panel, pg["ref_months"])
    sh = confirmed_shifts(Z)
    tid = panel.keys["territory_id"].to_numpy()
    news = headline_matrices(panel)
    news["постановление о режиме ЧС"] = decree_matrix(panel)
    pd.set_option("display.width", 220)

    # 1. покрытие
    cov = []
    allmo = panel.keys.territory_id.nunique()
    for g, M in news.items():
        mo_level = pd.DataFrame(M, index=tid).groupby(level=0).max()
        cov.append({"news": g, "mo_with_news": int((mo_level.sum(1) > 0).sum()), "mo_total": allmo,
                    "mo_months_with_news": int((mo_level.to_numpy() > 0).sum()),
                    "headlines_or_decrees": int(mo_level.to_numpy().sum())})
    cov = pd.DataFrame(cov)
    cov.to_csv(out_path(cfg, "news_coverage.csv"), index=False, encoding="utf-8")
    print(cov.to_string(index=False))

    # 2. реакция трат
    resp = []
    for g, M in news.items():
        for within in (False, True):
            r = response(panel, full, M, pg["boot"], rng, within)
            if r:
                resp.append({"news": g, **r})
    resp = pd.DataFrame(resp)
    resp.to_csv(out_path(cfg, "news_response.csv"), index=False, encoding="utf-8")
    print("\nСдвиг к своей категории в месяцы с новостью минус контроль, %:")
    print(resp.round(2).to_string(index=False))

    # 3. опережение
    # ряды МО, о которых в источнике хоть раз была новость: сравнение внутри них не путает новость с размером МО
    covered = {g: np.isin(tid, tid[(M > 0).any(1)]) for g, M in news.items()}
    es = []
    for g, M in news.items():
        for scope, rows in (("все ряды", full), ("МО, о которых пишут", full & covered[g])):
            if (M[rows] > 0).sum() < 30:
                continue
            e = event_study(sh[rows], M[rows], pg["shift_threshold"], 1.0, tid[rows], boot=pg["boot"], seed=cfg["seed"])
            e.insert(0, "scope", scope)
            e.insert(0, "news", g)
            es.append(e)
    es = pd.concat(es, ignore_index=True)
    es.to_csv(out_path(cfg, "news_event_study.csv"), index=False, encoding="utf-8")
    print("\nПодъём вероятности подтверждённого сдвига рядом с новостью (лаг < 0 - новость раньше сдвига):")
    print(es.pivot(index=["scope", "news"], columns="lag_news_vs_shift", values="lift").round(2).to_string())
    print(es[es.lag_news_vs_shift.isin([-2, -1])][["scope", "news", "lag_news_vs_shift", "lift", "ci_low", "ci_high"]]
          .round(2).to_string(index=False))

    # 4. детектор при равной доле тревог
    hits = {f"ssm_lr + {g}": M > 0 for g, M in news.items() if g in cfg["news_align"]["detector_news"]}
    det, S_lr, idx = detector_with_hits(panel, Z, full, sh, pg["detector_shift_threshold"],
                                        cfg["detection"]["news_factor"], pg["alarm_rates"], hits, return_scores=True)
    det.insert(0, "scope", "все ряды")
    dets = [det]
    for g in ("все местные заголовки", "постановление о режиме ЧС"):    # те же пороги, но только МО, о которых пишут
        d2 = detector_with_hits(panel, Z, full & covered[g], sh, pg["detector_shift_threshold"],
                                cfg["detection"]["news_factor"], pg["alarm_rates"], {f"ssm_lr + {g}": news[g] > 0})
        d2.insert(0, "scope", f"МО, о которых пишут: {g}")
        dets.append(d2)
    det = pd.concat(dets, ignore_index=True)
    det.to_csv(out_path(cfg, "news_detector.csv"), index=False, encoding="utf-8")
    print("\nДетектор при одинаковой доле месяцев с тревогой (метки - подтверждённые сдвиги):")
    print(det.pivot_table(index=["scope", "detector"], columns="alarm_rate", values=["precision", "recall"]).round(3).to_string())

    # 5. объяснимость тревог
    warm, T = 6, panel.T
    cols = np.arange(warm, T - 2)
    thr = np.quantile(S_lr[:, cols], 1 - cfg["news_align"]["explain_rate"])
    alarm = np.zeros_like(S_lr, bool)
    alarm[:, cols] = S_lr[:, cols] > thr
    exp_rows = []
    for g, M in news.items():
        recent = np.zeros_like(alarm)
        Mi = M[idx] > 0
        for t in range(T):
            recent[:, t] = Mi[:, max(0, t - 2):t + 1].any(1)
        base = np.zeros_like(alarm)
        base[:, cols] = True
        for scope, rows in (("все ряды", np.ones(len(idx), bool)), ("МО, о которых пишут", covered[g][idx])):
            a, b = alarm & rows[:, None], base & rows[:, None]
            exp_rows.append({"news": g, "scope": scope, "alarms": int(a.sum()),
                             "share_alarms_with_news": recent[a].mean(),
                             "share_random_months_with_news": recent[b].mean(),
                             "ratio": recent[a].mean() / max(recent[b].mean(), 1e-9)})
    ex = pd.DataFrame(exp_rows)
    ex.to_csv(out_path(cfg, "news_explain.csv"), index=False, encoding="utf-8")
    print(f"\nДоля тревог ssm_lr (самые сильные {cfg['news_align']['explain_rate']:.0%} ряд-месяцев) с местной новостью "
          "в тот же месяц или за 2 месяца до:")
    print(ex.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
