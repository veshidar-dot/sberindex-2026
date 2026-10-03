"""Разведочный анализ: разложение log-расходов на уровень МО, национальную компоненту и локальный остаток."""
import numpy as np
import pandas as pd

DATA = "data/hackathonlicence/consumption.parquet"


def main():
    c = pd.read_parquet(DATA)
    c["y"] = np.log(c["value"])
    c["n"] = c.groupby(["category", "date"])["y"].transform("mean")
    c["d"] = c["y"] - c["n"]
    c["a"] = c.groupby(["territory_id", "category"])["d"].transform("mean")
    c["e"] = c["d"] - c["a"]
    for cat, g in c.groupby("category"):
        tot = g["y"].var()
        print(f"{cat:22s} уровень={g['a'].var()/tot:.3f} нац.={g['n'].var()/tot:.3f} "
              f"остаток={g['e'].var()/tot:.3f} sd(e)={g['e'].std():.3f}")


if __name__ == "__main__":
    main()
