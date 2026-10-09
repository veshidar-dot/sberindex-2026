"""Прогноз вперёд от последнего месяца панели (декабрь 2024) на 12 месяцев 2025 года - тем же ансамблем,
что и в проверке: те же члены, те же параметры, равные веса в логах. Прогноз замораживается: файл и его
SHA-256 записываются в манифест, чтобы проверить его, когда СберИндекс опубликует данные МО за 2025 год.

    python -m sbi.forward
"""
import hashlib
import json
import time

import numpy as np
import pandas as pd

from .config import load_config, out_path
from .data import load_panel
from .forecast import get_model
from .national import month_after


def main():
    cfg = load_config()
    fw = cfg["forward"]
    panel = load_panel(cfg)
    o = panel.month_index(fw["origin"])
    H = fw["horizon"]
    members = cfg["models"]["ensemble"]["members"]
    logs = []
    for m in members:
        t = time.time()
        F = get_model(m)(panel, [o], H, dict(cfg["models"].get(m) or {}))
        np.savez_compressed(out_path(cfg, "forward", f"{m}.npz"), F=F, origins=np.array([o]))
        logs.append(np.log(F[0]))
        print(f"{m}: {time.time() - t:.0f} с, NaN {np.isnan(F).mean():.2%}", flush=True)
    E = np.exp(np.mean(logs, axis=0))
    months = [month_after(fw["origin"], h) for h in range(1, H + 1)]
    df = pd.DataFrame(E, columns=months)
    df.insert(0, "category", panel.keys["category"].to_numpy())
    df.insert(0, "territory_id", panel.keys["territory_id"].to_numpy())
    df = df.melt(id_vars=["territory_id", "category"], var_name="month", value_name="forecast")
    df = df.dropna().sort_values(["territory_id", "category", "month"])
    df["forecast"] = df.forecast.round(1)
    path = out_path(cfg, "forward", fw["file"])
    df.to_csv(path, index=False, encoding="utf-8", compression="gzip")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"origin": fw["origin"], "months": months, "members": members, "rows": len(df),
                "file": fw["file"], "sha256": sha, "created": time.strftime("%Y-%m-%dT%H:%M%z")}
    out_path(cfg, "forward", "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
