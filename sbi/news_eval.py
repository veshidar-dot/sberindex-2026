"""Эффект новостей на прогноз: MAE только на рядах МО из каталога событий и только для точек начала
от месяца события до 5 мес. после (когда новость уже известна и может помочь).

    python -m sbi.news_eval
"""
import numpy as np
import pandas as pd

from .config import load_config, out_path
from .data import load_panel
from .news import load_events
from .report import load_forecasts


def main():
    cfg = load_config()
    P = load_panel(cfg)
    ev = load_events(cfg)
    fc = load_forecasts(cfg, ["panel_ssm", "panel_ssm_news", "prophet", "naive"])
    tid = P.keys["territory_id"].to_numpy()
    rows = []
    for name, (F, origins) in fc.items():
        for _, e in ev.iterrows():
            t0 = P.month_index(e.event_month)
            idx = np.where(np.isin(tid, e.ids))[0]
            for k, o in enumerate(origins):
                if not (t0 <= o <= t0 + 5):
                    continue
                for h in (1, 3, 6):
                    if o + h >= P.T:
                        continue
                    y, f = P.V[idx, o + h], F[k, idx, h - 1]
                    ok = np.isfinite(y) & np.isfinite(f)
                    rows += [{"model": name, "event": e.event_id, "h": h, "abs_err": v}
                             for v in np.abs(y[ok] - f[ok])]
    d = pd.DataFrame(rows)
    t = d.groupby(["model", "h"]).abs_err.agg(["mean", "count"])
    t.reset_index().to_csv(out_path(cfg, "news_effect.csv"), index=False, encoding="utf-8")
    print(t.unstack("h").round(1).to_string())
    print(d.groupby(["event", "model"]).abs_err.mean().unstack().round(1).to_string())


if __name__ == "__main__":
    main()
