"""Метрики на схеме скользящего начала прогноза.

Прогноз модели хранится как массив F[o, i, h-1] (уровни, руб.): точка начала o
(индекс последнего известного месяца), ряд i, горизонт h. Цель - V[i, o+h].
"""
import numpy as np
import pandas as pd
from scipy import stats

from .data import CATEGORIES


def stack(panel, F, origins, h):
    """Пары (прогноз, факт) для горизонта h по всем точкам начала, где факт известен."""
    rows = []
    for k, o in enumerate(origins):
        t = o + h
        if t >= panel.T:
            continue
        rows.append(pd.DataFrame({
            "origin": o, "i": np.arange(panel.n), "cat": panel.cat,
            "y": panel.V[:, t], "f": F[k, :, h - 1]}))
    if not rows:
        return pd.DataFrame(columns=["origin", "i", "cat", "y", "f"])
    d = pd.concat(rows, ignore_index=True)
    return d[np.isfinite(d.y)]


def metrics(d):
    e = d.y - d.f
    ok = np.isfinite(e)
    e, y = e[ok], d.y[ok]
    sst = ((y - y.mean()) ** 2).sum()
    return {"MAE": np.abs(e).mean(),
            "R2": 1 - (e ** 2).sum() / sst if sst > 0 else np.nan,
            "WAPE": np.abs(e).sum() / np.abs(y).sum(),
            "N": int(ok.sum()), "coverage": ok.mean() if len(ok) else np.nan}


def table(panel, forecasts, origins, horizons):
    """Длинная таблица метрик: модель x горизонт x категория (+ 'Итого')."""
    out = []
    for name, F in forecasts.items():
        for h in horizons:
            d = stack(panel, F, origins, h)
            out.append({"model": name, "h": h, "category": "Итого", **metrics(d)})
            for c, g in d.groupby("cat"):
                out.append({"model": name, "h": h, "category": CATEGORIES[c], **metrics(g)})
    return pd.DataFrame(out)


def cumulative_mae(panel, F, origins, H, rows):
    """MAE в определении «горизонт H = средняя ошибка по шагам 1..H», только точки начала, у которых
    известны все H шагов, и только ряды rows (так считают другие работы конкурса)."""
    e = [np.abs(panel.V[rows, o + h] - F[k, rows, h - 1])
         for k, o in enumerate(origins) if o + H < panel.T for h in range(1, H + 1)]
    return float(np.nanmean(np.concatenate(e)))


def diebold_mariano(panel, F_a, F_b, origins, h):
    """DM-тест для |e|: средняя разность потерь по МО внутри каждой точки начала,
    затем HAC-дисперсия по ряду точек начала (лаг h-1). Если точка одна (h=12),
    то дисперсия по рядам с кластеризацией по МО: разности потерь МО суммируются по категориям."""
    da, db = stack(panel, F_a, origins, h), stack(panel, F_b, origins, h)
    m = da.merge(db[["origin", "i", "f"]], on=["origin", "i"], suffixes=("_a", "_b"))
    m = m[np.isfinite(m.f_a) & np.isfinite(m.f_b)]
    m["dl"] = np.abs(m.y - m.f_a) - np.abs(m.y - m.f_b)
    m["tid"] = panel.keys["territory_id"].to_numpy()[m.i]
    if m.origin.nunique() >= 4:
        s = m.groupby("origin").dl.mean().to_numpy()
        T = len(s)
        u = s - s.mean()
        lag = min(h - 1, T - 2)
        v = u @ u / T + 2 * sum((1 - L / (lag + 1)) * (u[L:] @ u[:-L]) / T for L in range(1, lag + 1))
        stat = s.mean() / np.sqrt(max(v, 1e-12) / T)
        p = 2 * stats.t.sf(abs(stat), df=T - 1)
        return {"mean_diff": m.dl.mean(), "DM": stat, "p": p, "units": T, "by": "origin"}
    g = m.groupby(["origin", "tid"]).dl.sum().to_numpy()
    stat = g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))
    return {"mean_diff": m.dl.mean(), "DM": stat, "p": 2 * stats.norm.sf(abs(stat)),
            "units": len(g), "by": "МО (кластеры)"}
