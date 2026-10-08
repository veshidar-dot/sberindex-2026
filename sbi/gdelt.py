"""Поток новостей GDELT по МО: справочник «МО - регион», неожиданность новостей и исследование
опережения («предупреждают ли новости о сдвиге заранее»).

    python -m sbi.gdelt

Данные: ref/news (агрегаты GDELT 2.0 по МО и регионам, см. ref/news/SOURCES.md).

Неожиданность новостей surprise[i, t] = log(1 + x[i, t]) - среднее log(1 + x[i, s]) за s = t-6 .. t-1,
где x - число сообщений о МО за месяц (всех, негативных или о бедствиях). Считается только по прошлому,
поэтому годится как признак в точке начала прогноза. Месяц без сообщений о МО - ноль. Для МО, о которых
GDELT не пишет никогда, используется поток их региона.

Подтверждённый сдвиг (задним числом, только для оценки): shift[i, t] = среднее z за t..t+2 минус среднее
за t-3..t-1, z - ряд без общей компоненты в единицах шума (как в детекции). Сдвиг подтверждён, если
|shift| >= порог. Подъём (lift) = P(сдвиг в t | новостной всплеск в t+k) / P(сдвиг в t);
k < 0 - новость пришла раньше сдвига в данных (предупреждение заранее).
"""
import argparse

import numpy as np
import pandas as pd

from .config import load_config, out_path, resolve
from .data import CATEGORIES, load_panel
from .detect_eval import prepare

HAZARD = ["g_disaster", "g_flood", "g_fire"]


def load_gdelt(level="mo"):
    d = pd.read_parquet(resolve(f"ref/news/gdelt_monthly_{level}.parquet"))
    d["hazard"] = d[HAZARD].sum(axis=1)
    return d


def mo_region(panel):
    """territory_id -> region_code: по сопоставлению мест GDELT, остальное - по соседним кодам.
    Коды МО СберИндекса идут блоками по регионам; МО между двумя МО одного региона получает этот
    регион, иначе - регион ближайшего по коду МО с известным регионом."""
    m = pd.read_csv(resolve("ref/news/gdelt_location_match.csv")).dropna(subset=["territory_id", "region_code"])
    known = m.groupby(m.territory_id.astype(int)).region_code.agg(lambda s: s.mode().iloc[0]).astype(int)
    kid = np.array(sorted(known.index))
    rows = []
    for t in np.sort(panel.keys["territory_id"].unique()):
        if t in known.index:
            rows.append((t, known[t], "gdelt"))
            continue
        j = np.searchsorted(kid, t)
        lo = kid[j - 1] if j > 0 else None
        hi = kid[j] if j < len(kid) else None
        if lo is not None and hi is not None and known[lo] == known[hi]:
            rows.append((t, known[lo], "между МО одного региона"))
        else:
            near = min([x for x in (lo, hi) if x is not None], key=lambda x: abs(x - t))
            rows.append((t, known[near], "ближайший код"))
    return pd.DataFrame(rows, columns=["territory_id", "region_code", "source"])


def _surprise(d, key, months, var, window):
    p = d.pivot_table(index=key, columns="month", values=var, aggfunc="sum").reindex(columns=months)
    arr = np.log1p(p.fillna(0.0).to_numpy())
    s = np.full(arr.shape, np.nan)
    for t in range(window, arr.shape[1]):
        s[:, t] = arr[:, t] - arr[:, t - window:t].mean(1)
    return pd.DataFrame(s, index=p.index.astype(int), columns=months)


def surprise_matrix(panel, var="hazard", window=6):
    """Неожиданность новостей для каждого ряда панели (МО x категория) по месяцам панели.
    own[i] - есть ли у МО собственные сообщения в GDELT (иначе взят поток региона)."""
    pre = [f"2022-{m:02d}" for m in range(1, 13)]
    months = pre + panel.months
    mo_s = _surprise(load_gdelt("mo"), "territory_id", months, var, window)[panel.months]
    rg_s = _surprise(load_gdelt("region"), "region_code", months, var, window)[panel.months]
    reg = mo_region(panel).set_index("territory_id").region_code
    tid = panel.keys["territory_id"].to_numpy()
    S = np.full((panel.n, panel.T), np.nan)
    own = np.isin(tid, mo_s.index)
    S[own] = mo_s.loc[tid[own]].to_numpy()
    r = reg.reindex(tid).to_numpy()
    use = ~own & np.isin(r, rg_s.index)
    S[use] = rg_s.loc[r[use]].to_numpy()
    return S, own


def confirmed_shifts(Z, w=3):
    N, T = Z.shape
    sh = np.full((N, T), np.nan)
    for t in range(w, T - w + 1):
        sh[:, t] = np.nanmean(Z[:, t:t + w], 1) - np.nanmean(Z[:, t - w:t], 1)
    return sh


def event_study(sh, news, thr_shift, thr_news, groups, lags=(-2, -1, 0, 1, 2), boot=500, seed=0):
    """Подъём вероятности подтверждённого сдвига рядом с новостным всплеском.
    Ожидаемое число сдвигов считается по месяцам (доля сдвигов среди всех рядов в тот же месяц), чтобы
    совпадение новостей и сдвигов по календарю не выдавалось за связь: lift = наблюдаемые / ожидаемые.
    Бутстреп по группам (МО для собственных новостей, регион для региональных)."""
    N, T = sh.shape
    rng = np.random.default_rng(seed)
    big = np.abs(sh) >= thr_shift
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(ug))]
    rows = []
    for k in lags:
        src = np.clip(np.arange(T) + k, 0, T - 1)
        inside = (np.arange(T) + k >= 0) & (np.arange(T) + k < T)
        nv = news[:, src]
        ok = np.isfinite(sh) & np.isfinite(nv) & inside[None, :]
        nw = ok & (np.nan_to_num(nv, nan=-np.inf) >= thr_news)
        def lift(idx):
            o, n_, b = ok[idx], nw[idx], big[idx]
            if not n_.any():
                return np.nan, 0, np.nan, np.nan
            p_month = (b & o).sum(0) / np.maximum(o.sum(0), 1)     # в месяце без наблюдений n_ тоже пуст
            expected = (n_ * p_month[None, :]).sum()
            observed = (b & n_).sum()
            return observed / expected if expected > 0 else np.nan, int(n_.sum()), b[n_].mean(), expected / n_.sum()
        est, n_ev, p_news, p_all = lift(np.arange(N))
        bs = [lift(np.concatenate([members[g] for g in rng.integers(0, len(ug), len(ug))]))[0] for _ in range(boot)]
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        rows.append({"lag_news_vs_shift": k, "lift": est, "ci_low": lo, "ci_high": hi,
                     "p_shift_given_news": p_news, "p_shift_expected": p_all, "news_months": n_ev})
    return pd.DataFrame(rows)


def detector_with_news(panel, Z, full, sh, thr_shift, thr_news, factor, rates, window=3, warm=6):
    """ssm_lr онлайн на полных рядах против ssm_lr + априорные лог-шансы 2 ln(factor) в месяцы, когда
    у МО (или его региона) был новостной всплеск за последние window месяцев. Пороги подобраны так, что
    доля месяцев с тревогой одинакова (rates). Метки - подтверждённые задним числом сдвиги |shift| >= thr_shift:
    тревога в t верна, если сдвиг начался в t-window+1 .. t; сдвиг пойман, если тревога пришла в t .. t+window-1."""
    from .detect import ssm_lr
    from .models.panel_ssm import fit
    idx = np.where(full)[0]
    S_lr = np.zeros((len(idx), panel.T))
    for c in range(6):
        m = panel.cat[idx] == c
        lam, phi = fit(Z[idx[m]])[:2]
        S_lr[m] = ssm_lr(Z[idx[m]], lam, phi, w=window, warm=warm)
    out = []
    for var in ("hazard", "docs", "neg_docs"):
        N_, own = surprise_matrix(panel, var)
        N_ = N_[idx]
        hit = np.nan_to_num(N_, nan=-np.inf) >= thr_news
        prior = np.zeros_like(S_lr)
        for t in range(panel.T):
            prior[:, t] = 2 * np.log(factor) * hit[:, max(0, t - window + 1):t + 1].any(1)
        big = np.abs(np.nan_to_num(sh[idx])) >= thr_shift
        T = panel.T
        cols = np.arange(warm, T - 2)          # месяцы, где метка определена
        for name, S in (("ssm_lr", S_lr), (f"ssm_lr + GDELT {var}", S_lr + prior)):
            vals = S[:, cols]
            for r in rates:
                thr = np.quantile(vals, 1 - r)
                alarm = np.zeros_like(S, bool)
                alarm[:, cols] = vals > thr
                ok_alarm = np.zeros_like(alarm)
                caught = np.zeros_like(big)
                for t in cols:
                    lo = max(0, t - window + 1)
                    ok_alarm[:, t] = alarm[:, t] & big[:, lo:t + 1].any(1)
                    caught[:, t] = big[:, t] & alarm[:, t:t + window].any(1)
                nb = big[:, cols].sum()
                out.append({"news": var, "detector": name, "alarm_rate": r,
                            "precision": ok_alarm.sum() / max(alarm.sum(), 1),
                            "recall": caught[:, cols].sum() / max(nb, 1),
                            "alarms": int(alarm.sum()), "shifts": int(nb)})
    return pd.DataFrame(out).drop_duplicates(subset=["detector", "alarm_rate"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    pg = cfg["gdelt"]
    panel = load_panel(cfg)
    reg = mo_region(panel)
    reg.to_csv(out_path(cfg, "mo_region.csv"), index=False, encoding="utf-8")
    print("МО -> регион:", reg.source.value_counts().to_dict(), "регионов", reg.region_code.nunique())

    Z, _, full = prepare(panel, pg["ref_months"])
    sh = confirmed_shifts(Z)
    tid = panel.keys["territory_id"].to_numpy()
    rcode = reg.set_index("territory_id").region_code.reindex(tid).to_numpy()
    res = []
    for var in pg["vars"]:
        S, own = surprise_matrix(panel, var)
        for scope, m in (("свои новости МО", full & own), ("новости региона", full & ~own)):
            for cat in [None] + list(range(6)):
                mm = m & (panel.cat == cat) if cat is not None else m
                if mm.sum() < 50:
                    continue
                grp = tid[mm] if scope == "свои новости МО" else rcode[mm]
                es = event_study(sh[mm], S[mm], pg["shift_threshold"], pg["news_threshold"], grp,
                                 boot=pg["boot"], seed=cfg["seed"])
                es.insert(0, "category", "все" if cat is None else CATEGORIES[cat])
                es.insert(0, "scope", scope)
                es.insert(0, "news", var)
                res.append(es)
    res = pd.concat(res, ignore_index=True)
    res.to_csv(out_path(cfg, "gdelt_event_study.csv"), index=False, encoding="utf-8")
    pd.set_option("display.width", 220)
    print(res[res.category == "все"].round(3).to_string(index=False))

    det = detector_with_news(panel, Z, full, sh, pg["detector_shift_threshold"], pg["news_threshold"],
                             cfg["detection"]["news_factor"], pg["alarm_rates"])
    det.to_csv(out_path(cfg, "gdelt_detector.csv"), index=False, encoding="utf-8")
    print(det.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
