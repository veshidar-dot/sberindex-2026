"""Проверка вне выборки главного компонента модели - прогноза общей траектории (sbi.national).

Национальный ряд СберИндекса «Потребительские расходы» известен до 2026-09, а правила прогноза выбирались
по проверке 2023-12 ... 2024-11 и по истории до 2023 г. (sbi.studies). Здесь те же правила прогнозируют
сам национальный ряд (продовольствие, непродовольственные товары, общепит, услуги, итог) от каждой точки
начала двух периодов: проверочного (как на панели) и 2025-2026 гг., данные которых при выборе правил
не использовались. Ошибка - |log прогноз - log факт| x 100 (~ % ошибки), ровно на шаге h.
Сравнение: наивная, сезонная наивная, Prophet с годовой сезонностью на всей истории ряда (с 2018-12).

    python -m sbi.national_oos
"""
import logging
import warnings

import numpy as np
import pandas as pd

from .config import load_config, out_path
from .national import RULES, forecast_national, load_external

COLS = {"total": 0, "food": 1, "nonfood": 2, "catering": 3, "services": 5}   # c, для которой EXT_MAP[c] = ряд


def _prophet(s, months, o, H):
    from prophet import Prophet
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    df = pd.DataFrame({"ds": pd.to_datetime(months[:o + 1]), "y": s[:o + 1]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False).fit(df)
    fut = m.make_future_dataframe(periods=H, freq="MS")
    return m.predict(fut).yhat.to_numpy()[-H:]


def main():
    cfg = load_config()
    p = cfg["national_oos"]
    ext = load_external(cfg)
    months = list(ext.index)
    H = cfg["validation"]["max_h"]
    rows = []
    for period, (a, b) in p["periods"].items():
        for o in range(months.index(a), months.index(b) + 1):
            for col, c in COLS.items():
                s = ext[col].to_numpy()
                f = forecast_national(s, months, o, c, ext, H)
                f["naive"] = np.full(H, s[o])
                f["seasonal_naive"] = np.array([s[o + h - 12] for h in range(1, H + 1)])
                f["prophet"] = _prophet(s, months, o, H)
                for h in range(1, H + 1):
                    if o + h >= len(s):
                        break
                    for rule, v in f.items():
                        rows.append({"period": period, "origin": months[o], "series": col, "rule": rule,
                                     "h": h, "err": 100 * abs(v[h - 1] - s[o + h])})
        print(f"{period}: готово", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(out_path(cfg, "national_oos.csv"), index=False, encoding="utf-8")
    hs = cfg["validation"]["horizons"]
    t = (d[d.h.isin(hs)].groupby(["period", "rule", "h"]).err.mean().unstack("h").round(3))
    n = d[d.h.isin(hs)].groupby(["period", "h"]).origin.nunique().unstack("h")
    t.to_csv(out_path(cfg, "national_oos_summary.csv"), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(t.to_string())
    print("точек начала:\n", n.to_string())
    order = list(RULES) + ["combo", "naive", "seasonal_naive", "prophet"]
    print(t.reset_index().pivot(index="rule", columns="period").reindex(order).round(2).to_string())


if __name__ == "__main__":
    main()
