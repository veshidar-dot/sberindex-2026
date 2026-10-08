"""Разложение с собственной сезонностью МО (третий член ансамбля).

    log y_it = a_i + n_ct + z_it,  прогноз  z(T+h) = rho^h z(T) + (1 - rho^h) z(T+h-12)

a_i, n_ct - двусторонние эффекты по данным до точки начала; n_ct прогнозируется так же, как в
panel_ssm (sbi.national). Локальное отклонение от уровня МО затухает от текущего значения к
прошлогоднему отклонению в тот же месяц: на коротком горизонте важен последний месяц, на длинном -
сезонный профиль самого МО (курорты, вахтовые посёлки, студенческие города), которого нет в
национальной сезонности. Если год назад значения нет - только затухание.
"""
import numpy as np

from ..config import load_config
from ..national import forecast_national, load_external, two_way


def panel_seasonal(panel, origins, H, params=None):
    params = params or {}
    rho = params.get("rho", 0.6)
    rule = params.get("national_rule", "combo")
    ext = load_external(load_config())
    F = np.full((len(origins), panel.n, H), np.nan)
    hs = np.arange(1, H + 1)
    for c in range(6):
        rows = np.where(panel.cat == c)[0]
        for k, o in enumerate(origins):
            Yc = panel.Y[rows, :o + 1]
            a, n = two_way(Yc)
            z = Yc - n - a[:, None]
            last = np.array([r[np.isfinite(r)][-1] if np.isfinite(r).any() else 0.0 for r in z])
            nf = forecast_national(n, panel.months, o, c, ext, H, {"ext_map": params.get("ext_map", "fixed")})[rule]
            for h in hs:
                s = o + h - 12
                zs = z[:, s] if s >= 0 else np.full(len(rows), np.nan)
                w = rho ** h
                zl = np.where(np.isfinite(zs), w * last + (1 - w) * zs, w * last)
                F[k, rows, h - 1] = np.exp(a + nf[h - 1] + zl)
    return F
