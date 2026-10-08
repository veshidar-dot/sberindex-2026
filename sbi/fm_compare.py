"""Сравнение моделей на одних и тех же рядах: для моделей, посчитанных на подвыборке (TiRex на CPU),
MAE остальных моделей пересчитывается только на этих рядах.

    python -m sbi.fm_compare
"""
import numpy as np
import pandas as pd

from .config import load_config, out_path
from .data import load_panel
from .evaluate import diebold_mariano, table
from .report import load_forecasts

MODELS = ["tirex_local", "timesfm_local", "chronos2_local", "chronos2_ft_local", "ensemble", "prophet", "prophet_seasonal"]


def main():
    cfg = load_config()
    P = load_panel(cfg)
    fc = load_forecasts(cfg, MODELS)
    names = [m for m in MODELS if m in fc]
    O = fc[names[0]][1]
    rows = np.ones(P.n, bool)
    for m in names:
        rows &= np.isfinite(fc[m][0]).any(axis=(0, 2))
    print(f"общих рядов: {rows.sum()} из {P.n}; модели: {names}")
    F = {m: np.where(rows[None, :, None], fc[m][0], np.nan) for m in names}
    t = table(P, F, O, cfg["validation"]["horizons"])
    t.to_csv(out_path(cfg, "fm_compare.csv"), index=False, encoding="utf-8")
    pd.set_option("display.width", 200)
    print(t[t.category == "Итого"].pivot(index="model", columns="h", values="MAE").round(1).to_string())
    if "tirex_local" in F and "timesfm_local" in F:
        for h in cfg["validation"]["horizons"]:
            r = diebold_mariano(P, F["tirex_local"], F["timesfm_local"], O, h)
            print(f"TiRex против TimesFM, h = {h}: разность {r['mean_diff']:.1f} руб., p = {r['p']:.3f}")


if __name__ == "__main__":
    main()
