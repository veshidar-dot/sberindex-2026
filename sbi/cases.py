"""Галерея реальных случаев: что видели прогноз и детектор и что писали новости.

    python -m sbi.cases            # случаи из config.yaml (cases.items) -> out/cases.json
    python -m sbi.cases --scan     # кандидаты: ряды событий каталога и самые сильные тревоги, первые тревоги каналов

Для каждого ряда «МО x категория»: факт, прогнозы ансамбля и Prophet из точки начала перед событием (или перед
тревогой), помесячная статистика LR-теста ступеньки в трёх каналах - сам ряд, он же с новостью, ряд относительно
10 соседей по дорогам - и пороги из sbi.detect_real (5 % рядов выборки с тревогой за весь период). Детектор
работает онлайн: статистика в месяц t считается только по данным до t. Отдельный случай - национальный ряд
(сдвиг на уровне страны) с прогнозом правил общей траектории от декабря 2024 г.
"""
import argparse
import json

import numpy as np
import pandas as pd

from .config import data_path, load_config, out_path
from .data import CATEGORIES, load_panel, origins as get_origins
from .detect import DETECTORS
from .detect_eval import prepare, prepare_relative
from .models.panel_ssm import fit
from .national import forecast_national, load_external
from .national_oos import COLS
from .news import boost_matrix, load_events
from .report import load_forecasts

CHANNELS = ("ssm_lr", "ssm_lr_news", "ssm_lr_spatial")


def channel_scores(cfg, panel, rows):
    """Статистика LR-теста трёх каналов для рядов rows (только полные ряды) - ровно как в sbi.detect_real:
    масштаб шума в LR оценивается по всей пачке рядов, поэтому считаем на той же пачке (случайная выборка
    полных рядов + ряды событий, тот же seed), к которой добавлены запрошенные ряды."""
    p = cfg["detection"]
    start = p["real"]["ref_months"]
    Z0, _, full = prepare(panel, start)
    Zr0, _, nbcnt = prepare_relative(panel, cfg, start, p.get("neighbors", 10))
    events = load_events(cfg)
    rng = np.random.default_rng(cfg["seed"])
    tid = panel.keys["territory_id"].to_numpy()
    ev_rows = np.where(full & np.isin(tid, np.concatenate(events.ids.to_numpy())))[0]
    sample = rng.choice(np.where(full)[0], p["real"]["sample"], replace=False)
    batch = np.unique(np.r_[sample, ev_rows, rows])
    pos = np.searchsorted(batch, rows)
    cat = panel.cat[batch]
    SB = np.zeros((len(batch), panel.T))
    SrB = np.zeros_like(SB)
    for c in range(6):
        m = cat == c
        lam, phi = fit(Z0[full & (panel.cat == c)])[:2]
        SB[m] = DETECTORS["ssm_lr"](Z0[batch[m]], lam, phi, w=p["window"], warm=start)
        lam, phi = fit(Zr0[full & (panel.cat == c) & (nbcnt > 0)])[:2]
        Zr = np.where(nbcnt[batch[m], None] > 0, Zr0[batch[m]], Z0[batch[m]])
        SrB[m] = DETECTORS["ssm_lr"](Zr, lam, phi, w=p["window"], warm=start)
    S, Sr = SB[pos], SrB[pos]
    news = boost_matrix(panel, events, p["news_factor"])[rows]
    prior = np.zeros_like(S)
    for t in range(panel.T):
        prior[:, t] = 2 * np.log(news[:, max(0, t - p["window"] + 1):t + 1].max(1))
    S[:, :start] = Sr[:, :start] = np.nan                   # опорный период: детектор ещё не работает
    return {"ssm_lr": S, "ssm_lr_news": S + prior, "ssm_lr_spatial": Sr}, start


def first_alarm(s, thr, start):
    hit = np.where(np.nan_to_num(s[start:], nan=-np.inf) > thr)[0]
    return int(start + hit[0]) if len(hit) else None


def row_of(panel, tid, category):
    i = np.where((panel.keys.territory_id.to_numpy() == tid) & (panel.keys.category.to_numpy() == category))[0]
    return int(i[0]) if len(i) else None


def shift_rel(panel, i, t0):
    """Сдвиг ряда относительно своей категории: среднее лога за t0..t0+2 минус за t0-3..t0-1."""
    c = panel.cat == panel.cat[i]
    own = np.nanmean(panel.Y[i, t0:t0 + 3]) - np.nanmean(panel.Y[i, t0 - 3:t0])
    common = np.nanmean(panel.Y[c][:, t0:t0 + 3]) - np.nanmean(panel.Y[c][:, t0 - 3:t0])
    return float(own - common)


def scan(cfg, panel):
    thr = pd.read_csv(out_path(cfg, "detection_real_thresholds.csv"), index_col=0).iloc[:, 0]
    ev = pd.read_csv(out_path(cfg, "detection_real_events.csv"))
    top = pd.read_csv(out_path(cfg, "detection_real_top.csv"))
    cand = pd.concat([ev[["territory_id", "category", "month"]].assign(src="событие"),
                      top[["territory_id", "category", "month"]].assign(src="тревога")]).drop_duplicates()
    cand["row"] = [row_of(panel, t, c) for t, c in zip(cand.territory_id, cand.category)]
    cand = cand.dropna(subset=["row"])
    rows = cand.row.astype(int).to_numpy()
    S, start = channel_scores(cfg, panel, rows)
    names = pd.read_csv(data_path(cfg, "mo_names")).set_index("territory_id").name
    for ch in CHANNELS:
        cand[ch] = [panel.months[a] if (a := first_alarm(S[ch][j], thr[ch], start)) is not None else ""
                    for j in range(len(rows))]
    cand["shift"] = [shift_rel(panel, r, panel.month_index(m)) for r, m in zip(rows, cand.month)]
    cand["МО"] = names.reindex(cand.territory_id).to_numpy()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    print(cand.sort_values("shift").round(3).to_string(index=False))


def build(cfg, panel):
    thr = pd.read_csv(out_path(cfg, "detection_real_thresholds.csv"), index_col=0).iloc[:, 0]
    names = pd.read_csv(data_path(cfg, "mo_names")).set_index("territory_id").name
    events = load_events(cfg).set_index("event_id")
    fc = load_forecasts(cfg, ["ensemble", "prophet"])
    O = get_origins(cfg, panel)
    items = cfg["cases"]["items"]
    rows = np.array([row_of(panel, it["territory_id"], it["category"]) for it in items])
    S, start = channel_scores(cfg, panel, rows)
    out = []
    for j, it in enumerate(items):
        i = rows[j]
        ev = events.loc[it["event"]] if it.get("event") else None
        t0 = panel.month_index(ev.event_month if ev is not None else it["month"])
        o = min(max(t0 - 1, O[0]), O[-1])                     # точка начала: последний месяц до события
        k = O.index(o)
        h = np.arange(1, min(12, panel.T - 1 - o) + 1)
        f = {m: fc[m][0][k, i, :len(h)].tolist() for m in fc}
        after = slice(o + 1, o + 1 + min(3, len(h)))
        err = {m: float(np.mean(np.abs(np.array(f[m][:3]) - panel.V[i, after]))) for m in f}
        alarms = {ch: first_alarm(S[ch][j], thr[ch], start) for ch in CHANNELS}
        out.append({
            "id": it["id"], "title": it["title"], "kind": it["kind"],
            "territory_id": int(it["territory_id"]), "mo": names[it["territory_id"]], "category": it["category"],
            "event": None if ev is None else {"id": it["event"], "title": ev.title, "month": ev.event_month,
                                              "announce_date": str(ev.announce_date.date()), "source": ev.source},
            "month": panel.months[t0], "origin": panel.months[o],
            "actual": panel.V[i].tolist(), "forecast": f, "mae_3m": err,
            "shift_log": shift_rel(panel, i, t0),
            "scores": {ch: S[ch][j].tolist() for ch in CHANNELS},
            "thresholds": {ch: float(thr[ch]) for ch in CHANNELS},
            "alarm": {ch: (panel.months[a] if a is not None else None) for ch, a in alarms.items()},
            "alarm_months": {ch: [panel.months[t] for t in range(start, panel.T)
                                  if np.isfinite(S[ch][j, t]) and S[ch][j, t] > thr[ch]] for ch in CHANNELS},
            "lag": {ch: (a - t0 if a is not None else None) for ch, a in alarms.items()},
        })
        print(f"{it['id']}: {names[it['territory_id']]}, {it['category']}, сдвиг {out[-1]['shift_log']:+.3f}, "
              f"тревоги {out[-1]['alarm']}, MAE 3 мес. {({m: round(v) for m, v in err.items()})}", flush=True)

    # сдвиг на уровне страны: национальный ряд и прогноз правил общей траектории от декабря 2024 г.
    nat = cfg["cases"]["national"]
    ext = load_external(cfg)
    months = list(ext.index)
    col = nat["series"]
    o = months.index(nat["origin"])
    s = ext[col].to_numpy()
    fn = forecast_national(s, months, o, COLS[col], ext, 12)["combo"]
    lo = months.index(nat["from"])
    national = {"title": nat["title"], "series": col, "origin": nat["origin"], "months": months[lo:],
                "actual": np.exp(s[lo:]).tolist(),
                "forecast": {"months": months[o + 1:o + 13], "values": np.exp(fn).tolist()}}
    res = {"months": panel.months, "cases": out, "national": national}
    out_path(cfg, "cases.json").write_text(json.dumps(res, ensure_ascii=False, default=float), encoding="utf-8")
    print("national:", col, "прогноз на", nat["origin"], "->", national["forecast"]["months"][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    panel = load_panel(cfg)
    scan(cfg, panel) if a.scan else build(cfg, panel)


if __name__ == "__main__":
    main()
