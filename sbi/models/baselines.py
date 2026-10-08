"""Базовые модели. Каждая функция: (panel, origins, H, params) -> F[o, i, h-1], уровни в руб.
В точке начала o модель видит только V[:, :o+1]."""
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd


def last_observed(V):
    """Последнее ненулевое значение каждого ряда."""
    idx = np.where(np.isfinite(V), np.arange(V.shape[1]), -1).max(axis=1)
    out = np.full(V.shape[0], np.nan)
    ok = idx >= 0
    out[ok] = V[np.where(ok)[0], idx[ok]]
    return out


def naive(panel, origins, H, params=None):
    F = np.empty((len(origins), panel.n, H))
    for k, o in enumerate(origins):
        F[k] = last_observed(panel.V[:, :o + 1])[:, None]
    return F


def seasonal_naive(panel, origins, H, params=None):
    """y[o+h] = y[o+h-12]; если значения нет - последнее известное."""
    F = np.empty((len(origins), panel.n, H))
    for k, o in enumerate(origins):
        last = last_observed(panel.V[:, :o + 1])
        for h in range(1, H + 1):
            s = o + h - 12
            v = panel.V[:, s] if 0 <= s <= o else np.full(panel.n, np.nan)
            F[k, :, h - 1] = np.where(np.isfinite(v), v, last)
    return F


# ---------- Prophet: отдельная модель на каждый ряд, как в постановке задачи ----------

def _prophet_chunk(args):
    """seas - None или (S_hist, S_fut): сезонные факторы (в логах) по точкам начала, формы
    (точки, ряды, месяцы до o включительно) и (точки, ряды, H). С ними Prophet подгоняется на
    y / exp(S_hist) - ряд без сезонности, тренд как у обычного Prophet (линейный, в рублях), -
    а прогноз умножается обратно: yhat * exp(S_fut)."""
    rows, dates, origins, H, params, seas = args
    logging.getLogger("cmdstanpy").disabled = True
    logging.getLogger("prophet").disabled = True
    from prophet import Prophet
    out = np.full((len(origins), len(rows), H), np.nan)
    failed = 0
    for j, y in enumerate(rows):
        for k, o in enumerate(origins):
            yy = y[:o + 1] if seas is None else y[:o + 1] / np.exp(seas[0][k][j])
            d = pd.DataFrame({"ds": dates[:o + 1], "y": yy}).dropna()
            back = (lambda v: v) if seas is None else (lambda v, k=k, j=j: v * np.exp(seas[1][k][j]))
            if len(d) < 3:
                out[k, j] = back(np.repeat(d.y.iloc[-1], H)) if len(d) else np.nan
                continue
            try:
                m = Prophet(**params).fit(d)
                fut = pd.DataFrame({"ds": [dates[o] + pd.DateOffset(months=h) for h in range(1, H + 1)]})
                out[k, j] = back(m.predict(fut)["yhat"].to_numpy())
            except RuntimeError:        # оптимизатор Stan не сошёлся: последнее значение
                out[k, j] = back(np.repeat(d.y.iloc[-1], H))
                failed += 1
    return out, failed


def _seasonal_adjustment(panel, origins, H, rows):
    """Сезонные факторы национального ряда категории (по данным до точки начала) для рядов rows."""
    from ..config import load_config
    from ..national import EXT_MAP, load_external, month_after, seasonal_factors
    ext = load_external(load_config())
    hist, fut = [], []
    for o in origins:
        sf = {c: seasonal_factors(ext[EXT_MAP[c]].loc[:panel.months[o]], panel.months[o]) for c in range(6)}
        cats = panel.cat[rows]
        hist.append(np.array([[sf[c][m[5:]] for m in panel.months[:o + 1]] for c in cats]))
        fut.append(np.array([[sf[c][month_after(panel.months[o], h)[5:]] for h in range(1, H + 1)] for c in cats]))
    return hist, fut


def prophet(panel, origins, H, params=None):
    """Каждый кусок рядов сохраняется в out/cache/<name>: после сбоя расчёт продолжается.
    seasonal_profile: true - Prophet на ряде, очищенном от сезонных факторов национального ряда категории."""
    from ..config import load_config, out_path
    params = dict(params or {})
    n_jobs = params.pop("n_jobs", os.cpu_count() - 2)
    profile = params.pop("seasonal_profile", False)
    name = params.pop("cache_name", "prophet")
    cache = out_path(load_config(), "cache", name, "x").parent
    dates = panel.dates()
    chunks = np.array_split(np.arange(panel.n), n_jobs * 8)
    F = np.empty((len(origins), panel.n, H))
    todo = []
    for i, c in enumerate(chunks):
        f = cache / f"chunk_{i:04d}_{len(chunks)}.npy"
        if f.exists():
            F[:, c] = np.load(f)
        else:
            todo.append((i, c, f))
    failed = 0
    with ProcessPoolExecutor(n_jobs) as ex:
        futs = {ex.submit(_prophet_chunk, (panel.V[c], dates, origins, H, params,
                                           _seasonal_adjustment(panel, origins, H, c) if profile else None)): (i, c, f)
                for i, c, f in todo}
        for done, fu in enumerate(as_completed(futs), 1):
            i, c, f = futs[fu]
            res, nf = fu.result()
            np.save(f, res)
            F[:, c] = res
            failed += nf
            if done % 20 == 0:
                print(f"  prophet: {done}/{len(todo)} кусков", flush=True)
    total = len(origins) * panel.n
    print(f"  prophet: неудачных подгонок {failed} из {total} ({failed / total:.3%})", flush=True)
    return F
