"""Панель расходов: ряды (МО x категория) x месяцы, пропуски - NaN."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import data_path

CATEGORIES = ["Все категории", "Продовольствие", "Маркетплейсы",
              "Общественное питание", "Здоровье", "Транспорт"]


@dataclass
class Panel:
    keys: pd.DataFrame   # territory_id, category, cat (индекс категории) - по строке на ряд
    months: list         # ['2023-01', ...]
    V: np.ndarray        # уровни, руб. на жителя, shape (n, T)

    @property
    def Y(self):
        return np.log(self.V)

    @property
    def n(self):
        return self.V.shape[0]

    @property
    def T(self):
        return self.V.shape[1]

    @property
    def cat(self):
        return self.keys["cat"].to_numpy()

    def month_index(self, ym):
        return self.months.index(ym)

    def dates(self):
        return pd.to_datetime(self.months, format="%Y-%m")


def load_panel(cfg):
    c = pd.read_parquet(data_path(cfg, "consumption"))
    w = c.pivot_table(index=["territory_id", "category"], columns="date",
                      values="value", aggfunc="first")
    w = w.reindex(columns=sorted(w.columns))
    keys = w.index.to_frame(index=False)
    keys["cat"] = keys["category"].map({k: i for i, k in enumerate(CATEGORIES)}).astype(int)
    order = np.lexsort((keys["territory_id"].to_numpy(), keys["cat"].to_numpy()))
    keys = keys.iloc[order].reset_index(drop=True)
    V = w.to_numpy(dtype=float)[order]
    V[V <= 0] = np.nan
    return Panel(keys=keys, months=list(w.columns), V=V)


def origins(cfg, panel):
    v = cfg["validation"]
    a, b = panel.month_index(v["first_origin"]), panel.month_index(v["last_origin"])
    return list(range(a, b + 1))
