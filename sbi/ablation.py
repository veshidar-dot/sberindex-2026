"""Откуда берётся ошибка: общая компонента категории против локальной части МО.

    python -m sbi.ablation

Локальная часть фиксирована и проста (уровень МО + последнее отклонение * rho^h), меняется только
прогноз общей компоненты n_ct: правила из sbi.national, их среднее и «оракул» - фактическое
будущее n_ct (недостижимая граница: показывает, сколько ошибки вносит общая часть).
"""
import numpy as np
import pandas as pd

from .config import load_config, out_path
from .data import load_panel, origins as get_origins
from .evaluate import table
from .national import RULES, forecast_national, load_external, two_way


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext-map", default="fixed", choices=["fixed", "auto"])
    args = ap.parse_args()
    cfg = load_config()
    P = load_panel(cfg)
    O = get_origins(cfg, P)
    H = cfg["validation"]["max_h"]
    ext = load_external(cfg)
    rho = 0.6
    names = list(RULES) + ["combo", "random_walk", "oracle"]
    res = {k: np.full((len(O), P.n, H), np.nan) for k in names}
    hs = np.arange(1, H + 1)
    for c in range(6):
        m = P.cat == c
        _, n_full = two_way(P.Y[m])
        for k, o in enumerate(O):
            a, n = two_way(P.Y[m, :o + 1])
            z = P.Y[m, :o + 1] - n - a[:, None]
            zT = np.array([r[np.isfinite(r)][-1] if np.isfinite(r).any() else 0.0 for r in z])
            nf = forecast_national(n, P.months, o, c, ext, H, {"ext_map": args.ext_map})
            nf["random_walk"] = np.repeat(n[o], H)
            nf["oracle"] = np.array([n[o] + n_full[o + h] - n_full[o] if o + h < P.T else np.nan for h in hs])
            started = np.isfinite(z).any(1)                  # у ряда без наблюдений до o прогноза нет
            for nm in names:
                res[nm][k][m] = np.where(started[:, None],
                                         np.exp(a[:, None] + nf[nm][None, :] + rho ** hs[None, :] * zT[:, None]), np.nan)
    t = table(P, res, O, cfg["validation"]["horizons"])
    t.to_csv(out_path(cfg, f"ablation_national_{args.ext_map}.csv"), index=False, encoding="utf-8")
    print(t[t.category == "Итого"].pivot(index="model", columns="h", values="MAE").round(0).to_string())


if __name__ == "__main__":
    main()
