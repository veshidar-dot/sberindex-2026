"""Граф МО по автодорогам: k ближайших соседей (connection.parquet хранит пары в одну сторону)."""
import numpy as np
import pandas as pd

from .config import data_path


def nearest_neighbors(cfg, ids, k=10, kind="highway"):
    c = pd.read_parquet(data_path(cfg, "connection"), filters=[("type", "==", kind)],
                        columns=["territory_id_x", "territory_id_y", "distance"])
    ids = set(int(x) for x in ids)
    c = c[c.territory_id_x.isin(ids) & c.territory_id_y.isin(ids)]
    both = pd.concat([c.rename(columns={"territory_id_x": "a", "territory_id_y": "b"}),
                      c.rename(columns={"territory_id_x": "b", "territory_id_y": "a"})])
    both = both[both.a != both.b].sort_values(["a", "distance"])
    nn = both.groupby("a").head(k)
    return nn.groupby("a").b.apply(np.array).to_dict()


def market_access(cfg):
    m = pd.read_parquet(data_path(cfg, "market_access"))
    return m.set_index("territory_id").market_access
