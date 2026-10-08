"""Прогноз общей (национальной) компоненты n_ct по категории.

n_ct - временной эффект в разложении log y_it = a_i + n_ct + r_it (двусторонние эффекты).
Год панели не позволяет оценить сезонность, поэтому используем национальный ряд
СберИндекса «Потребительские расходы» (с 2018-12): только наблюдения до точки начала o.

Правила (все в логах, возвращают n_hat[h-1], h = 1..H):
  snaive  - n[o+h-12] + G                                (сезонная наивная с дрейфом)
  hybrid  - n[o] + сезонная разность + G*h/12           (разность из панели за прошлый год, иначе из нац. ряда)
  airline - n[o] + (x_hat[o+h] - x[o]), x - нац. ряд, SARIMA(0,1,1)(0,1,1)12
  extlvl  - уровень без сезонности (3 мес.) + G*h/12 + сезонный фактор нац. ряда
  combo   - среднее четырёх правил
Какой национальный ряд соответствует категории, выбирается в каждой точке начала по данным до неё
(ext_map: auto - минимум RMSE месячных изменений) или задаётся содержательно (ext_map: fixed, EXT_MAP,
основной вариант: выбран до оценки на тесте).
G - годовой рост: по нац. ряду за последние 12 мес. (среднее 3 мес. к тем же месяцам год назад);
если в панели уже есть год назад, усредняется с ростом панели.
"""
import warnings

import numpy as np
import pandas as pd

from .config import data_path

EXT_MAP = {0: "total", 1: "food", 2: "nonfood", 3: "catering", 4: "total", 5: "services"}
RULES = ("snaive", "hybrid", "airline", "extlvl")


def load_external(cfg):
    x = pd.read_csv(data_path(cfg, "national")).set_index("date")
    x["total"] = x[["food", "nonfood", "catering", "services"]].sum(axis=1)
    return np.log(x)


def two_way(Yc, iters=30):
    """Двусторонние эффекты y_it = a_i + n_t на несбалансированной панели (чередующиеся средние)."""
    n = np.nanmean(Yc, axis=0)
    a = np.zeros(Yc.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(iters):
            a = np.nanmean(Yc - n, axis=1)
            a = np.where(np.isfinite(a), a, 0.0)
            n = np.nanmean(Yc - a[:, None], axis=0)
    shift = n.mean()
    return a + shift, n - shift


def seasonal_factors(s, upto, skip=("2020",)):
    """Сезонные факторы ряда s (лог) по классической схеме 2x12 MA, медиана по годам."""
    s = s.loc[:upto]
    ma = s.rolling(12, center=True).mean().rolling(2).mean().shift(-1)
    dev = (s - ma).dropna()
    dev = dev[~dev.index.str[:4].isin(skip)]
    f = dev.groupby(dev.index.str[5:]).median()
    return f - f.mean()


def _airline(s, H):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = SARIMAX(s, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
    return r.forecast(H)


def choose_external(n, months, o, ext):
    """Национальный ряд, месячные изменения которого ближе всего (RMSE) к изменениям n
    по данным до точки начала o включительно."""
    x = ext.loc[months[0]:months[o]]
    dn = np.diff(n[:o + 1])
    err = {col: np.sqrt(np.mean((dn - np.diff(x[col].to_numpy())) ** 2))
           for col in ("total", "food", "nonfood", "catering", "services")}
    return min(err, key=err.get)


def month_after(ym, h):
    y, m = int(ym[:4]), int(ym[5:])
    k = y * 12 + m - 1 + h
    return f"{k // 12}-{k % 12 + 1:02d}"


def forecast_national(n, months, o, c, ext, H, params=None):
    """Все правила для одной категории c и точки начала o. n - временной эффект до o включительно."""
    params = params or {}
    col = choose_external(n, months, o, ext) if params.get("ext_map", "fixed") == "auto" else EXT_MAP[c]
    xs = ext[col].loc[:months[o]]
    T = months[o]
    sf = seasonal_factors(xs, T)
    i = len(xs) - 1
    # годовой рост нац. ряда: последние 3 мес. к тем же месяцам год назад ("last", по умолчанию),
    # средний с 2019 г. по 12-месячным суммам ("longrun") или их среднее ("half"). "half" лучше на
    # истории самого нац. ряда (рост 2022-2023 гг.), но на панели не помог - оставлен как вариант.
    g_last = xs.iloc[i - 2:i + 1].mean() - xs.iloc[i - 14:i - 11].mean()
    s12 = np.log(np.exp(xs).rolling(12).sum())
    base = s12.loc["2019-12"]
    g_long = (s12.iloc[-1] - base) / ((len(s12.loc["2019-12":]) - 1) / 12)
    g_ext = {"last": g_last, "longrun": g_long, "half": (g_last + g_long) / 2}[params.get("growth", "last")]
    G = np.mean([g_ext, n[o - 2:o + 1].mean() - n[o - 14:o - 11].mean()]) if o >= 14 else g_ext

    out = {}
    hs = np.arange(1, H + 1)
    tgt = [month_after(T, h) for h in hs]
    out["snaive"] = np.array([n[o + h - 12] + G if 0 <= o + h - 12 <= o else n[o] + G * h / 12 for h in hs])
    # сезонная разность T -> T+h: из панели за прошлый год (очищена от прошлогоднего роста
    # n[o] - n[o-12], иначе рост учитывался бы дважды) или из сезонных факторов нац. ряда
    if o - 12 >= 0:
        g_last = n[o] - n[o - 12]
        ds = np.array([n[o + h - 12] - n[o - 12] - g_last * h / 12 if o + h - 12 <= o
                       else sf[m[5:]] - sf[T[5:]] for h, m in zip(hs, tgt)])
    else:
        ds = np.array([sf[m[5:]] - sf[T[5:]] for m in tgt])
    out["hybrid"] = n[o] + ds + G * hs / 12
    start = params.get("airline_start", "2021-01")
    xf = _airline(xs.loc[start:].to_numpy(), H)
    out["airline"] = n[o] + (xf - xs.iloc[-1])
    d = n[:o + 1] - np.array([sf[m[5:]] for m in months[:o + 1]])
    lvl = d[-3:].mean() + G / 12
    out["extlvl"] = np.array([lvl + G * h / 12 + sf[m[5:]] for h, m in zip(hs, tgt)])
    out["combo"] = np.mean([out[r] for r in RULES], axis=0)
    if params.get("chronos"):
        from .models.chronos_fm import median_forecast, pipeline
        xc = median_forecast(pipeline(), [xs.to_numpy().astype(np.float32)], H)[0]
        out["chronos"] = n[o] + (xc - xs.iloc[-1])
    return out
