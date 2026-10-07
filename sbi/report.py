"""Сводит сохранённые прогнозы в таблицу метрик и тесты DM против базовой модели.

    python -m sbi.report
"""
import argparse

import numpy as np
import pandas as pd

from .config import load_config, out_path
from .data import load_panel
from .evaluate import diebold_mariano, table


def load_forecasts(cfg, names=None):
    d = out_path(cfg, "forecasts", "x").parent
    out = {}
    for f in sorted(d.glob("*.npz")):
        if names and f.stem not in names:
            continue
        z = np.load(f)
        out[f.stem] = (z["F"], list(z["origins"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    panel = load_panel(cfg)
    fc = load_forecasts(cfg, a.models)
    hs = cfg["validation"]["horizons"]
    origins = next(iter(fc.values()))[1]
    assert all(o == origins for _, o in fc.values()), "разные точки начала у моделей"
    tab = table(panel, {k: F for k, (F, _) in fc.items()}, origins, hs)
    tab.to_csv(out_path(cfg, "metrics.csv"), index=False, encoding="utf-8")
    pd.set_option("display.width", 200)
    tot = tab[tab.category == "Итого"].pivot(index="model", columns="h", values=["MAE", "R2"])
    print(tot.round(3).to_string())

    base = cfg["validation"]["baseline"]
    if base in fc:
        rows = []
        for name, (F, _) in fc.items():
            if name == base:
                continue
            for h in hs:
                rows.append({"model": name, "h": h, **diebold_mariano(panel, F, fc[base][0], origins, h)})
        dm = pd.DataFrame(rows)
        dm.to_csv(out_path(cfg, "dm_vs_baseline.csv"), index=False, encoding="utf-8")
        print(f"\nDM против {base} (mean_diff < 0 - модель точнее):")
        print(dm.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
