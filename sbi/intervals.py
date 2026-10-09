"""Интервалы прогноза и их покрытие.

    python -m sbi.intervals

Интервал строится вокруг прогноза ансамбля (в логах) из двух частей неопределённости, обе только по данным
до точки начала:
  1. общая траектория категории - средний квадрат ошибки правила combo (sbi.national) на самом национальном ряде
     СберИндекса для того же горизонта h по точкам начала с intervals.nat_from до o - h (ошибка известна к o);
  2. местная часть МО - дисперсия прогноза модели пространства состояний (раздел 5 отчёта):
     s k_i [P_mm + 2 phi^h P_mr + phi^2h P_rr + lam h + (1 - phi^2h) / (1 - phi^2)].
Части складываются, интервал - нормальный в логах. Ничего не подбирается по тесту.
Сравнение - собственные интервалы Prophet (квантили 500 выборок из апостериорного прогноза) на стратифицированной
подвыборке рядов: покрытие 80 % и 95 % интервалов, средняя ширина и интервальная оценка Винклера (95 %).
"""
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

from .config import load_config, out_path
from .data import CATEGORIES, load_panel, origins as get_origins
from .models.panel_ssm import fit, kalman, series_scale
from .national import EXT_MAP, forecast_national, load_external, two_way
from .report import load_forecasts

LEVELS = (0.80, 0.95)


def national_mse(cfg, ext, H):
    """err2[col][месяц o'][h-1] - квадрат ошибки combo на национальном ряде col из точки o'."""
    months = list(ext.index)
    a = months.index(cfg["intervals"]["nat_from"])
    out = {}
    for col in sorted(set(EXT_MAP.values())):
        c = next(k for k, v in EXT_MAP.items() if v == col)
        s = ext[col].to_numpy()
        e = {}
        for o in range(a, len(s) - 1):
            f = forecast_national(s, months, o, c, ext, H)["combo"]
            e[months[o]] = np.array([(f[h - 1] - s[o + h]) ** 2 if o + h < len(s) else np.nan for h in range(1, H + 1)])
        out[col] = e
    return out


def local_var(Yc, H):
    """Дисперсия прогноза локальной части на 1..H шагов для рядов категории (логи)."""
    a, n = two_way(Yc)
    Z = Yc - n
    kap = series_scale(Z - a[:, None])
    lam, phi, _ = fit(Z / np.sqrt(kap)[:, None])
    _, _, P, s, _, _ = kalman(Z / np.sqrt(kap)[:, None], lam, phi)
    h = np.arange(1, H + 1)
    pv = P[:, 0, 0][:, None] + 2 * phi ** h * P[:, 0, 1][:, None] + phi ** (2 * h) * P[:, 1, 1][:, None]
    acc = lam * h + (1 - phi ** (2 * h)) / max(1 - phi ** 2, 1e-6)
    return s * kap[:, None] * (pv + acc[None, :])


def our_intervals(cfg, panel, O, H):
    """sd[k, i, h-1] - стандартное отклонение прогноза в логах."""
    ext = load_external(cfg)
    err2 = national_mse(cfg, ext, H)
    sd = np.full((len(O), panel.n, H), np.nan)
    for k, o in enumerate(O):
        for c in range(6):
            rows = np.where(panel.cat == c)[0]
            vl = local_var(panel.Y[rows, :o + 1], H)
            e = err2[EXT_MAP[c]]
            known = [m for m in e if m <= panel.months[o]]
            # ошибка из точки o' на шаге h известна к o, только если o' + h <= o
            vn = np.array([np.nanmean([e[m][h - 1] for m in known[:len(known) - h]]) for h in range(1, H + 1)])
            sd[k, rows] = np.sqrt(vl + vn[None, :])
        print(f"  интервалы: точка {panel.months[o]}", flush=True)
    return sd


def _prophet_rows(args):
    rows, dates, origins, H, n_samples = args
    logging.getLogger("cmdstanpy").disabled = True
    logging.getLogger("prophet").disabled = True
    from prophet import Prophet
    q = [(1 - l) / 2 for l in LEVELS] + [(1 + l) / 2 for l in LEVELS]
    out = np.full((len(origins), len(rows), H, len(q)), np.nan)
    for j, y in enumerate(rows):
        for k, o in enumerate(origins):
            d = pd.DataFrame({"ds": dates[:o + 1], "y": y[:o + 1]}).dropna()
            if len(d) < 3:
                continue
            try:
                m = Prophet(yearly_seasonality="auto", weekly_seasonality=False, daily_seasonality=False,
                            uncertainty_samples=n_samples).fit(d)
                fut = pd.DataFrame({"ds": [dates[o] + pd.DateOffset(months=h) for h in range(1, H + 1)]})
                smp = m.predictive_samples(fut)["yhat"]          # (H, выборки)
                out[k, j] = np.quantile(smp, q, axis=1).T
            except RuntimeError:
                pass
    return out


def prophet_intervals(cfg, panel, O, H, rows):
    p = cfg["intervals"]
    chunks = np.array_split(rows, p["n_jobs"] * 4)
    dates = panel.dates()
    res = np.full((len(O), panel.n, H, 2 * len(LEVELS)), np.nan)
    with ProcessPoolExecutor(p["n_jobs"]) as ex:
        for c, r in zip(chunks, ex.map(_prophet_rows, [(panel.V[c], dates, O, H, p["prophet_samples"]) for c in chunks])):
            res[:, c] = r
    return res


def score(y, lo, hi, level):
    """Покрытие, средняя ширина (руб.) и интервальная оценка Винклера (руб.)."""
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    y, lo, hi = y[ok], lo[ok], hi[ok]
    a = 1 - level
    w = hi - lo
    ws = w + 2 / a * (lo - y) * (y < lo) + 2 / a * (y - hi) * (y > hi)
    return {"coverage": float(((y >= lo) & (y <= hi)).mean()), "width": float(w.mean()), "winkler": float(ws.mean()),
            "N": int(ok.sum())}


def main():
    cfg = load_config()
    p = cfg["intervals"]
    panel = load_panel(cfg)
    O = get_origins(cfg, panel)
    H = cfg["validation"]["max_h"]
    hs = cfg["validation"]["horizons"]
    F = load_forecasts(cfg, ["ensemble"])["ensemble"][0]
    sd = our_intervals(cfg, panel, O, H)
    np.savez_compressed(out_path(cfg, "intervals_ensemble.npz"), sd=sd)

    rng = np.random.default_rng(cfg["seed"])
    sub = np.concatenate([rng.choice(np.where((panel.cat == c) & np.isfinite(panel.V).all(1))[0],
                                     p["prophet_per_category"], replace=False) for c in range(6)])
    f = out_path(cfg, "intervals_prophet.npz")
    if f.exists():
        PQ = np.load(f)["q"]
    else:
        PQ = prophet_intervals(cfg, panel, O, H, sub)
        np.savez_compressed(f, q=PQ)

    rows = []
    for h in hs:
        ks =[k for k, o in enumerate(O) if o + h < panel.T]
        y = np.stack([panel.V[:, O[k] + h] for k in ks])
        cats = np.broadcast_to(panel.cat, y.shape)
        insub = np.zeros(panel.n, bool)
        insub[sub] = True
        insub = np.broadcast_to(insub, y.shape)
        for j, level in enumerate(LEVELS):
            z = stats.norm.ppf((1 + level) / 2)
            c = np.log(F[ks, :, h - 1])
            lo, hi = np.exp(c - z * sd[ks, :, h - 1]), np.exp(c + z * sd[ks, :, h - 1])
            plo, phi_ = PQ[ks, :, h - 1, j], PQ[ks, :, h - 1, len(LEVELS) + j]
            for cat in ["Итого"] + CATEGORIES:
                m = np.ones_like(y, bool) if cat == "Итого" else cats == CATEGORIES.index(cat)
                rows.append({"model": "ensemble", "rows": "все", "h": h, "level": level, "category": cat,
                             **score(y[m], lo[m], hi[m], level)})
                ms = m & insub
                rows.append({"model": "ensemble", "rows": "подвыборка", "h": h, "level": level, "category": cat,
                             **score(y[ms], lo[ms], hi[ms], level)})
                rows.append({"model": "prophet", "rows": "подвыборка", "h": h, "level": level, "category": cat,
                             **score(y[ms], plo[ms], phi_[ms], level)})
    t = pd.DataFrame(rows)
    t.to_csv(out_path(cfg, "intervals.csv"), index=False, encoding="utf-8")
    pd.set_option("display.width", 220)
    tot = t[t.category == "Итого"]
    print(tot.pivot_table(index=["model", "rows", "level"], columns="h", values=["coverage", "winkler"]).round(3).to_string())
    print(t[(t.level == 0.8) & (t.rows == "все")].pivot(index="category", columns="h", values="coverage").round(2).to_string())


if __name__ == "__main__":
    main()
