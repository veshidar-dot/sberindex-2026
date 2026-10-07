"""Базовые модели. Каждая функция: (panel, origins, H, params) -> F[o, i, h-1], уровни в руб.
В точке начала o модель видит только V[:, :o+1]."""
import logging
import os
from concurrent.futures import ProcessPoolExecutor

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
    rows, dates, origins, H, params = args
    logging.getLogger("cmdstanpy").disabled = True
    logging.getLogger("prophet").disabled = True
    from prophet import Prophet
    out = np.full((len(origins), len(rows), H), np.nan)
    for j, y in enumerate(rows):
        for k, o in enumerate(origins):
            d = pd.DataFrame({"ds": dates[:o + 1], "y": y[:o + 1]}).dropna()
            if len(d) < 3:
                out[k, j] = d.y.iloc[-1] if len(d) else np.nan
                continue
            m = Prophet(**params).fit(d)
            fut = pd.DataFrame({"ds": [dates[o] + pd.DateOffset(months=h) for h in range(1, H + 1)]})
            out[k, j] = m.predict(fut)["yhat"].to_numpy()
    return out


def prophet(panel, origins, H, params=None):
    params = dict(params or {})
    n_jobs = params.pop("n_jobs", os.cpu_count() - 2)
    dates = panel.dates()
    chunks = np.array_split(np.arange(panel.n), n_jobs * 8)
    tasks = [(panel.V[c], dates, origins, H, params) for c in chunks]
    F = np.empty((len(origins), panel.n, H))
    with ProcessPoolExecutor(n_jobs) as ex:
        for c, res in zip(chunks, ex.map(_prophet_chunk, tasks)):
            F[:, c] = res
    return F
