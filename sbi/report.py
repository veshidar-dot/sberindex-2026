"""Сводит сохранённые прогнозы в таблицу метрик и тесты DM против базовой модели.

    python -m sbi.report
"""
import argparse

import numpy as np
import pandas as pd

from .config import load_config, out_path
from .data import load_panel
from .evaluate import cumulative_mae, diebold_mariano, table


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
    extra = cfg["validation"].get("extra_baselines", [])
    for b in [base] + extra:
        if b not in fc:
            continue
        rows = []
        for name, (F, _) in fc.items():
            if name == b or np.isnan(F).all(axis=(0, 2)).mean() > 0.5:   # модели на подвыборке - в sbi.fm_compare
                continue
            for h in hs:
                rows.append({"model": name, "h": h, **diebold_mariano(panel, F, fc[b][0], origins, h)})
        dm = pd.DataFrame(rows)
        dm.to_csv(out_path(cfg, "dm_vs_baseline.csv" if b == base else f"dm_vs_{b}.csv"), index=False, encoding="utf-8")
        print(f"\nDM против {b} (mean_diff < 0 - модель точнее):")
        print(dm[dm.model.isin(["ensemble", "panel_ssm", "timesfm_local", "prophet", "prophet_seasonal", "naive"])].round(4).to_string(index=False))
    # для сравнения с другими работами: средняя ошибка по шагам 1..H на полных рядах
    full = np.isfinite(panel.V).all(axis=1)
    cum = pd.DataFrame([{"model": name, "h": h, "MAE_1toH": cumulative_mae(panel, F, origins, h, full)}
                        for name, (F, _) in fc.items() if np.isnan(F).all(axis=(0, 2)).mean() <= 0.5 for h in hs])
    cum.to_csv(out_path(cfg, "metrics_cumulative.csv"), index=False, encoding="utf-8")
    print(f"\nMAE по шагам 1..H, {full.sum()} полных рядов:")
    print(cum.pivot(index="model", columns="h", values="MAE_1toH").round(1).to_string())
    write_markdown(cfg, tab, out_path(cfg, "dm_vs_baseline.csv") if base in fc else None, hs)


def write_markdown(cfg, tab, dm_path, hs):
    """Таблицы для отчёта (out/tables.md): итоговые метрики и MAE по категориям."""
    def md(df):
        cols = list(df.columns)
        lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
        return "\n".join(lines)
    tot = tab[tab.category == "Итого"]
    out = []
    for metric, f in (("MAE", lambda v: f"{v:,.0f}".replace(",", " ")), ("R2", lambda v: f"{v:.3f}"),
                      ("WAPE", lambda v: f"{v:.1%}")):
        # лучшие сверху: по MAE и WAPE - по возрастанию, по R2 - по убыванию
        p = tot.pivot(index="model", columns="h", values=metric).sort_values(hs[-1], ascending=metric != "R2")
        p = p.map(f).reset_index().rename(columns={h: f"h = {h}" for h in hs})
        out += [f"### {metric}", "", md(p), ""]
    for h in hs:
        p = tab[tab.h == h].pivot(index="model", columns="category", values="MAE").round(0).astype("Int64").reset_index()
        out += [f"### MAE по категориям, h = {h}", "", md(p), ""]
    if dm_path is not None:
        dm = pd.read_csv(dm_path)
        dm["p"] = dm.p.map(lambda v: "< 0.001" if v < 0.001 else f"{v:.3f}")
        dm["mean_diff"] = dm.mean_diff.round(1)
        dm["DM"] = dm.DM.round(2)
        out += ["### Диболд-Мариано против базовой модели", "", md(dm[["model", "h", "mean_diff", "DM", "p", "units", "by"]]), ""]
    out_path(cfg, "tables.md").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
