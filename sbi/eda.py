"""Разведочный анализ: доли дисперсии лога расходов - уровень МО, общая компонента месяца, остаток.

    python -m sbi.eda
"""
import numpy as np
import pandas as pd

from .config import data_path, load_config


def main():
    c = pd.read_parquet(data_path(load_config(), "consumption"))
    c["y"] = np.log(c["value"])
    c["n"] = c.groupby(["category", "date"])["y"].transform("mean")
    c["d"] = c["y"] - c["n"]
    c["a"] = c.groupby(["territory_id", "category"])["d"].transform("mean")
    c["e"] = c["d"] - c["a"]
    full = c.groupby(["territory_id", "category"]).date.transform("size") == 24
    for cat, g in c.groupby("category"):
        tot = g["y"].var()
        b = g[full[g.index]]
        lm = b.groupby(b.date.str[:4]).y.mean()
        print(f"{cat:22s} уровень={g['a'].var()/tot:.3f} общая={g['n'].var()/tot:.3f} "
              f"остаток={g['e'].var()/tot:.3f} sd(e)={g['e'].std():.3f} "
              f"рост 2024/2023 (геом., полные ряды)={np.exp(lm['2024'] - lm['2023']) - 1:+.1%}")


if __name__ == "__main__":
    main()
