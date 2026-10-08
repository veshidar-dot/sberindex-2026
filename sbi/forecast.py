"""Строит прогнозы выбранных моделей на всех точках начала и сохраняет их в out/forecasts/.

    python -m sbi.forecast --models naive seasonal_naive prophet
"""
import argparse
import importlib
import time

import numpy as np

from .config import load_config, out_path
from .data import load_panel, origins as get_origins

REGISTRY = {
    "naive": "sbi.models.baselines:naive",
    "seasonal_naive": "sbi.models.baselines:seasonal_naive",
    "prophet": "sbi.models.baselines:prophet",
    "panel_ssm": "sbi.models.panel_ssm:panel_ssm",
    "panel_ssm_chronos": "sbi.models.panel_ssm:panel_ssm",
    "panel_ssm_news": "sbi.models.panel_ssm:panel_ssm",
    "panel_ssm_automap": "sbi.models.panel_ssm:panel_ssm",
    "panel_seasonal": "sbi.models.panel_seasonal:panel_seasonal",
    "lgbm": "sbi.models.lgbm:lgbm",
    "lgbm_news": "sbi.models.lgbm:lgbm",
    "ensemble": "sbi.models.ensemble:ensemble",
    "chronos2": "sbi.models.chronos_fm:chronos2",
    "chronos2_cross": "sbi.models.chronos_fm:chronos2_cross",
    "chronos2_local": "sbi.models.fm_local:chronos2_local",
    "timesfm": "sbi.models.fm_local:timesfm",
    "timesfm_local": "sbi.models.fm_local:timesfm_local",
}


def get_model(name):
    mod, fn = REGISTRY[name].split(":")
    return getattr(importlib.import_module(mod), fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(REGISTRY))
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    np.random.seed(cfg["seed"])
    panel = load_panel(cfg)
    origins = get_origins(cfg, panel)
    H = cfg["validation"]["max_h"]
    for name in a.models:
        t = time.time()
        params = dict(cfg["models"].get(name) or {})
        F = get_model(name)(panel, origins, H, params)
        np.savez_compressed(out_path(cfg, "forecasts", f"{name}.npz"), F=F, origins=np.array(origins))
        if "_fitted" in params:
            import pandas as pd
            pd.DataFrame(params["_fitted"]).to_csv(out_path(cfg, "fitted", f"{name}.csv"), index=False)
        print(f"{name}: {F.shape}, {time.time() - t:.0f} с, NaN {np.isnan(F).mean():.2%}", flush=True)


if __name__ == "__main__":
    main()
