"""Собирает результаты в один JSON для интерактивного лендинга (landing/data.json).

    python -m sbi.export_landing
"""
import json

import numpy as np
import pandas as pd

from .config import data_path, load_config, out_path, resolve
from .data import CATEGORIES, load_panel
from .national import load_external, two_way
from .report import load_forecasts


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    return x


def main():
    cfg = load_config()
    P = load_panel(cfg)
    names = pd.read_csv(data_path(cfg, "mo_names")).set_index("territory_id").name
    out = {"months": P.months, "categories": CATEGORIES}

    m = pd.read_csv(out_path(cfg, "metrics.csv"))
    out["metrics"] = m.to_dict("records")
    dm = out_path(cfg, "dm_vs_baseline.csv")
    if dm.exists():
        out["dm"] = pd.read_csv(dm).to_dict("records")
    for k in ("ablation_national_fixed", "ablation_national_auto", "detection_synthetic",
              "detection_real_events", "detection_real_top"):
        f = out_path(cfg, f"{k}.csv")
        if f.exists():
            out[k] = pd.read_csv(f).to_dict("records")

    # общая компонента по категориям и национальный ряд
    ext = load_external(cfg)
    out["national_panel"] = {CATEGORIES[c]: two_way(P.Y[P.cat == c])[1].tolist() for c in range(6)}
    out["national_ext"] = {"months": ext.index.tolist(), **{c: ext[c].tolist() for c in ext.columns}}

    # примеры рядов: факт и прогнозы из точки 2023-12 (h=1..12) и 2024-03 (для событий весны 2024)
    fc = load_forecasts(cfg)
    show = [m for m in ("prophet", "panel_ssm", "panel_ssm_news", "lgbm", "chronos2", "naive") if m in fc]
    tid = P.keys["territory_id"].to_numpy()
    ex_ids = [1673, 1672, 1333, 2192, 785, 1459, 1665]
    examples = []
    for t in ex_ids:
        for c in range(6):
            i = np.where((tid == t) & (P.cat == c))[0]
            if not len(i):
                continue
            i = i[0]
            e = {"territory_id": t, "name": names[t], "category": CATEGORIES[c],
                 "actual": P.V[i].tolist(), "forecasts": {}}
            for mname in show:
                F, origins = fc[mname]
                e["forecasts"][mname] = {P.months[o]: F[k, i].tolist() for k, o in enumerate(origins)
                                         if P.months[o] in ("2023-12", "2024-03", "2024-06")}
            examples.append(e)
    out["examples"] = examples
    f = resolve("landing/data.json")
    f.parent.mkdir(exist_ok=True)
    f.write_text(json.dumps(_clean(out), ensure_ascii=False), encoding="utf-8")
    print(f, f.stat().st_size // 1024, "КБ")


if __name__ == "__main__":
    main()
