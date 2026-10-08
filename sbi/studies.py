"""Проверенные и отвергнутые гипотезы: региональная компонента и модель роста общей траектории.

    python -m sbi.studies

1. Регион. Справочник «МО - регион» - sbi.gdelt.mo_region. Считаем долю дисперсии локального остатка МО,
   которую объясняет «регион x месяц», и «оракула»: насколько улучшился бы прогноз, если бы будущее изменение
   средней по региону локальной компоненты было известно точно (верхняя граница пользы от региональной модели).
2. Рост. Годовой рост 12-месячной суммы розничного оборота (до 2018-11 - Росстат по описанию источника, далее ряд
   СберИндекса; ref/news/rosstat_retail_monthly.csv) прогнозируется на следующий год правилами и регрессиями
   с реальной ключевой ставкой (ref/key_rate.csv). Псевдо-реальное время: на каждом декабре 2016-2022 гг.
   регрессия оценивается только по годам, чей рост уже известен.
"""
import numpy as np
import pandas as pd

from .config import load_config, out_path, resolve
from .data import CATEGORIES, load_panel, origins as get_origins
from .evaluate import table
from .gdelt import mo_region
from .national import forecast_national, load_external, two_way


def region_study(cfg, panel):
    O = get_origins(cfg, panel)
    H = cfg["validation"]["max_h"]
    ext = load_external(cfg)
    r = mo_region(panel).set_index("territory_id").region_code.reindex(panel.keys.territory_id).to_numpy()
    share = []
    names = ["локальная часть сохраняется", "то же + оракул региона"]
    res = {k: np.full((len(O), panel.n, H), np.nan) for k in names}
    for c in range(6):
        m = np.where(panel.cat == c)[0]
        a_f, n_f = two_way(panel.Y[m])
        e = panel.Y[m] - n_f - a_f[:, None]
        R = pd.DataFrame(e).groupby(r[m]).transform("mean").to_numpy()
        share.append({"category": CATEGORIES[c], "share_region_month": np.nanvar(R) / np.nanvar(e)})
        for k, o in enumerate(O):
            a, n = two_way(panel.Y[m, :o + 1])
            z = panel.Y[m, :o + 1] - n - a[:, None]
            zT = np.array([q[np.isfinite(q)][-1] if np.isfinite(q).any() else 0.0 for q in z])
            nf = forecast_national(n, panel.months, o, c, ext, H)["combo"]
            for h in range(1, H + 1):
                base = a + nf[h - 1] + zT
                res[names[0]][k, m, h - 1] = np.exp(base)
                if o + h < panel.T:
                    res[names[1]][k, m, h - 1] = np.exp(base + R[:, o + h] - R[:, o])
    t = table(panel, res, O, cfg["validation"]["horizons"])
    return pd.DataFrame(share), t[t.category == "Итого"].pivot(index="model", columns="h", values="MAE")


def growth_study():
    r = pd.read_csv(resolve("ref/news/rosstat_retail_monthly.csv")).set_index("date")
    k = pd.read_csv(resolve("ref/key_rate.csv")).set_index("date")
    rows = []
    for col in ["retail_total", "retail_food", "retail_nonfood"]:
        s12 = np.log(r[col].dropna().rolling(12).sum())
        d = pd.DataFrame({"g_back": s12 - s12.shift(12), "g_fwd": s12.shift(-12) - s12})
        d = d.join(k[["key_rate", "real_key_rate"]], how="left")
        d["dkey6"] = d.key_rate - d.key_rate.shift(6)
        for Y in range(2017, 2024):
            o = f"{Y - 1}-12"
            if o not in d.index or not np.isfinite(d.loc[o, "g_fwd"]):
                continue
            x0, fact = d.loc[o], d.loc[o, "g_fwd"]
            train = d.loc[:f"{Y - 2}-12"]
            preds = {"рост за последний год": x0.g_back,
                     "средний рост за 5 лет": d.loc[:o].g_back.dropna().iloc[-60:].mean()}
            for name, cols in {"AR(1) годового роста": ["g_back"],
                               "AR(1) + реальная ставка": ["g_back", "real_key_rate"],
                               "AR(1) + реальная ставка + изменение ставки": ["g_back", "real_key_rate", "dkey6"]}.items():
                tr = train.dropna(subset=cols + ["g_fwd"])
                X = np.c_[np.ones(len(tr)), tr[cols].to_numpy()]
                b = np.linalg.lstsq(X, tr.g_fwd.to_numpy(), rcond=None)[0]
                preds[name] = float(np.r_[1, x0[cols].to_numpy(dtype=float)] @ b)
            for nm, p in preds.items():
                rows.append({"series": col, "year": Y, "rule": nm, "pred": p, "fact": fact, "abs_err": abs(p - fact)})
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    panel = load_panel(cfg)
    share, mae = region_study(cfg, panel)
    share.to_csv(out_path(cfg, "study_region_share.csv"), index=False, encoding="utf-8")
    mae.to_csv(out_path(cfg, "study_region_oracle.csv"), encoding="utf-8")
    g = growth_study()
    g.to_csv(out_path(cfg, "study_growth.csv"), index=False, encoding="utf-8")
    pd.set_option("display.width", 200)
    print(share.round(3).to_string(index=False))
    print(mae.round(0).to_string())
    print(g.pivot_table(index="rule", columns="series", values="abs_err", aggfunc="mean").round(3).to_string())
    print(g.groupby("rule").abs_err.mean().round(4).to_string())


if __name__ == "__main__":
    main()
