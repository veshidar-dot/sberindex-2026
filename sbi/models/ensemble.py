"""Ансамбль: среднее геометрическое (среднее в логах) сохранённых прогнозов моделей, веса равные и
заданы заранее (не подбираются по тесту). Члены ансамбля нужно построить раньше."""
import numpy as np

from ..config import load_config, out_path


def ensemble(panel, origins, H, params=None):
    members = (params or {}).get("members", ["panel_ssm", "lgbm"])
    cfg = load_config()
    logs = []
    for m in members:
        z = np.load(out_path(cfg, "forecasts", f"{m}.npz"))
        assert list(z["origins"]) == list(origins), f"{m}: другие точки начала"
        logs.append(np.log(z["F"]))
    return np.exp(np.mean(logs, axis=0))
