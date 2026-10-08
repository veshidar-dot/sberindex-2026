"""LightGBM на всей панели (прямая стратегия: один бустинг на все горизонты, h - признак).

Цель: log y[i, o+h] - log y[i, o] - (n_hat[c, o+h] - n[c, o]) - локальное изменение сверх
прогноза общей компоненты (того же, что в panel_ssm). Прогноз = y[i, o] * exp(n_hat - n + цель).
Для точки начала o обучение только на парах (o', h) с o' + h <= o: всё известно на момент o.
Признаки: h, месяц цели, категория, локальные отклонения z (o, o-1, o-2, o-12), изменения z
за 1 и 3 мес., уровень МО, индекс доступности рынков, средние изменения z у k соседей по дорогам;
в варианте lgbm_news - неожиданность новостей GDELT о МО (или его регионе) в месяцы o и o-1 (sbi.gdelt).
"""
import numpy as np
import pandas as pd

from ..config import load_config
from ..graph import market_access, nearest_neighbors
from ..national import forecast_national, load_external, two_way


def _features(panel, Z, a, o, hs, nb_mean, ma, news=None):
    """Признаки для точки начала o (данные до o) и горизонтов hs. Z - локальные отклонения.
    news - словарь «имя -> матрица неожиданности новостей (ряды x месяцы)»; берутся месяцы o и o-1,
    т. е. только новости, вышедшие к концу месяца точки начала."""
    n = panel.n
    def col(t):
        return Z[:, t] if t >= 0 else np.full(n, np.nan)
    z0, z1, z2, z12 = col(o), col(o - 1), col(o - 2), col(o - 12)
    nb1 = nb_mean(col(o) - col(o - 1))
    nb3 = nb_mean(col(o) - col(o - 3))
    base = {"cat": panel.cat, "z0": z0, "dz1": z0 - z1, "dz3": z0 - col(o - 3), "z1": z1, "z2": z2,
            "z12": z12, "level": a, "ma": ma, "nb_dz1": nb1, "nb_dz3": nb3,
            "nobs": np.isfinite(Z[:, :o + 1]).sum(1)}
    for name, M in (news or {}).items():
        base[f"{name}_0"] = M[:, o]
        base[f"{name}_1"] = M[:, o - 1] if o >= 1 else np.full(n, np.nan)
    frames = []
    for h in hs:
        f = pd.DataFrame(base)
        f["h"] = h
        f["month"] = (int(panel.months[o][5:]) - 1 + h) % 12 + 1
        frames.append(f)
    return frames


def lgbm(panel, origins, H, params=None):
    import lightgbm as lgb
    params = dict(params or {})
    rule = params.pop("national_rule", "combo")
    k = params.pop("neighbors", 10)
    news_vars = params.pop("news", [])
    cfg = load_config()
    ext = load_external(cfg)
    tid = panel.keys["territory_id"].to_numpy()
    news = None
    if news_vars:
        from ..gdelt import surprise_matrix
        news = {}
        for v in news_vars:
            S, own = surprise_matrix(panel, v)
            news[f"news_{v}"] = S
        news["news_own"] = np.repeat(own.astype(float)[:, None], panel.T, axis=1)
    nn = nearest_neighbors(cfg, np.unique(tid), k=k)
    ma = market_access(cfg).reindex(tid).to_numpy()
    # индекс строки (МО, категория) -> строки соседей той же категории
    pos = {(t, c): i for i, (t, c) in enumerate(zip(tid, panel.cat))}
    nb_rows = [np.array([pos[(b, c)] for b in nn.get(t, []) if (b, c) in pos], int)
               for t, c in zip(tid, panel.cat)]

    def nb_mean(x):
        return np.array([np.nanmean(x[r]) if len(r) and np.isfinite(x[r]).any() else np.nan
                         for r in nb_rows])

    F = np.full((len(origins), panel.n, H), np.nan)
    hs = np.arange(1, H + 1)
    for k_o, o in enumerate(origins):
        # общая компонента и локальные отклонения по данным до o
        Z = np.full((panel.n, o + 1), np.nan)
        a = np.zeros(panel.n)
        n_by_c, nf_by_c = {}, {}
        for c in range(6):
            r = panel.cat == c
            ac, nc = two_way(panel.Y[r, :o + 1])
            Z[r] = panel.Y[r, :o + 1] - nc - ac[:, None]
            a[r] = ac
            n_by_c[c] = nc
            nf_by_c[c] = forecast_national(nc, panel.months, o, c, ext, H)[rule]
        # обучающая выборка: начала o' < o, горизонты с o' + h <= o
        X_tr, y_tr = [], []
        for o2 in range(3, o):
            hh = [h for h in hs if o2 + h <= o]
            if not hh:
                continue
            for h, f in zip(hh, _features(panel, Z, a, o2, hh, nb_mean, ma, news)):
                dn = np.array([n_by_c[c][o2 + h] - n_by_c[c][o2] for c in panel.cat])
                y = panel.Y[:, o2 + h] - panel.Y[:, o2] - dn
                ok = np.isfinite(y) & np.isfinite(f.z0.to_numpy())
                X_tr.append(f[ok])
                y_tr.append(y[ok])
        X_tr, y_tr = pd.concat(X_tr), np.concatenate(y_tr)
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, categorical_feature=["cat"])
        last = np.array([panel.Y[i, :o + 1][np.isfinite(panel.Y[i, :o + 1])][-1]
                         if np.isfinite(panel.Y[i, :o + 1]).any() else np.nan for i in range(panel.n)])
        for h, f in zip(hs, _features(panel, Z, a, o, hs, nb_mean, ma, news)):
            dn = np.array([nf_by_c[c][h - 1] - n_by_c[c][o] for c in panel.cat])
            F[k_o, :, h - 1] = np.exp(last + dn + model.predict(f))
        print(f"  lgbm origin {panel.months[o]}: train {len(y_tr)}", flush=True)
    return F
