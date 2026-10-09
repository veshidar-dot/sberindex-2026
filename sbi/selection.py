"""Как выбирается лучшая модель и лучший детектор: не по одной цифре, а процедурой выбора.

1. Набор лучших моделей (Model Confidence Set; Hansen, Lunde, Nason, 2011): модели, которые на уровне alpha
   нельзя отвергнуть как худшие. Потери - |ошибка| на общих парах «точка начала x ряд», статистика T_max,
   бутстреп по МО (все категории и точки начала МО - один кластер). Вывод условен на проверочный период:
   временная зависимость между точками начала учтена в тестах Диболда-Мариано (sbi.report), а не здесь.
   Считается для каждого горизонта в целом и по категориям.
2. Сведение ранжирований. «Избиратели» - 24 клетки «горизонт x категория»; для детекторов - срезы синтетики
   (3 размера сдвига и 6 категорий) x 7 долей ложных тревог = 63 клетки, полнота в среднем по повторам,
   и для проверки устойчивости только 21 клетка «размер сдвига x доля тревог». Четыре правила: Борда (сумма мест),
   Коупленд (победы минус поражения в парных сравнениях большинством), Кемени (ранжирование с наименьшим
   суммарным числом парных расхождений с избирателями, точно - динамикой по подмножествам) и пороговое
   правило (Aleskerov, Chistyakov, Kalyagin, 2010: меньше худших мест лучше, при равенстве - меньше
   следующих за худшими и т. д.). Если победитель один при всех правилах, выбор не зависит от процедуры.

    python -m sbi.selection
"""
import itertools

import numpy as np
import pandas as pd

from .config import load_config, out_path
from .data import CATEGORIES, load_panel
from .report import load_forecasts


def losses(panel, fc, models, h, rows=None):
    """|ошибка| на парах, где известны факт и прогнозы всех моделей: (L[пары, модели], кластер МО, категория)."""
    L, tid, cat = [], [], []
    keys = panel.keys["territory_id"].to_numpy()
    for k, o in enumerate(next(iter(fc.values()))[1]):
        if o + h >= panel.T:
            continue
        y = panel.V[:, o + h]
        F = np.stack([fc[m][0][k, :, h - 1] for m in models], axis=1)
        ok = np.isfinite(y) & np.isfinite(F).all(1)
        if rows is not None:
            ok &= rows
        L.append(np.abs(F[ok] - y[ok, None]))
        tid.append(keys[ok])
        cat.append(panel.cat[ok])
    return np.concatenate(L), np.concatenate(tid), np.concatenate(cat)


def mcs(L, cluster, alpha, B, rng):
    """Набор лучших моделей по T_max с бутстрепом по кластерам. Возвращает p-значения MCS по моделям."""
    _, c = np.unique(cluster, return_inverse=True)
    C = c.max() + 1
    S = np.zeros((C, L.shape[1]))
    np.add.at(S, c, L)
    n = np.bincount(c, minlength=C).astype(float)
    d = S.sum(0) / n.sum()
    K = np.stack([np.bincount(rng.integers(0, C, C), minlength=C) for _ in range(B)]).astype(float)
    db = (K @ S) / (K @ n)[:, None]
    alive = list(range(L.shape[1]))
    p_mcs = np.ones(L.shape[1])
    prev = 0.0
    while len(alive) > 1:
        a = np.array(alive)
        dbar = d[a] - d[a].mean()
        dbb = db[:, a] - db[:, a].mean(1, keepdims=True)
        sd = np.sqrt(((dbb - dbar) ** 2).mean(0))
        t = dbar / sd
        tb = ((dbb - dbar) / sd).max(1)
        p = max(prev, float((tb >= t.max()).mean()))
        if p >= alpha:
            for m in alive:
                p_mcs[m] = p
            p_mcs[alive[int(np.argmin(d[a]))]] = 1.0
            break
        worst = alive[int(np.argmax(t))]
        p_mcs[worst] = p
        prev = p
        alive.remove(worst)
    return p_mcs, d


def ranks(score, higher_better=False):
    """Места (1 - лучшее) по столбцам score[клетки, альтернативы]."""
    s = -score if higher_better else score
    return s.argsort(1).argsort(1) + 1


def borda(R):
    return R.sum(0)                                   # меньше - лучше


def pairwise(R):
    """N[a, b] - число избирателей, у которых a выше b."""
    return (R[:, :, None] < R[:, None, :]).sum(0)


def copeland(R):
    N = pairwise(R)
    win = (N > N.T).sum(1)
    lose = (N < N.T).sum(1)
    return -(win - lose)                              # меньше - лучше


def kemeny(R):
    """Точное ранжирование Кемени: f(S) - наименьшее число расхождений, если S занимает верхние |S| мест."""
    N = pairwise(R)
    m = R.shape[1]
    full = 1 << m
    states = np.arange(full)
    add = np.zeros((m, full))                         # add[x, S] = сумма N[x, y] по y из S
    for y in range(m):
        bit = (states >> y) & 1
        add += N[:, y][:, None] * bit[None, :]
    f = np.full(full, np.inf)
    f[0] = 0
    arg = np.zeros(full, int)
    for S in range(full):
        if not np.isfinite(f[S]):
            continue
        for x in range(m):
            if S >> x & 1:
                continue
            T = S | (1 << x)
            v = f[S] + add[x, S]                      # x ставится под всеми из S: расходятся те, кто ставит x выше
            if v < f[T]:
                f[T], arg[T] = v, x
    order, S = [], full - 1
    while S:
        x = arg[S]
        order.append(x)
        S ^= 1 << x
    pos = np.empty(m, int)
    pos[order[::-1]] = np.arange(m)
    return pos                                        # 0 - первое место


def threshold(R):
    """Пороговое правило: лексикографически меньше худших мест."""
    m = R.shape[1]
    counts = np.stack([(R == g).sum(0) for g in range(m, 0, -1)], axis=1)   # [альтернатива, место m..1]
    order = sorted(range(m), key=lambda a: tuple(counts[a]))
    pos = np.empty(m, int)
    pos[order] = np.arange(m)
    return pos


def aggregate(R, names, kemeny_top=12):
    """Места по четырём правилам. Кемени - точно для первых kemeny_top по Борда (2^k состояний)."""
    out = pd.DataFrame({"model": names})
    out["Борда"] = pd.Series(borda(R)).rank(method="min").astype(int).to_numpy()
    out["Коупленд"] = pd.Series(copeland(R)).rank(method="min").astype(int).to_numpy()
    out["пороговое"] = threshold(R) + 1
    top = np.argsort(borda(R))[:kemeny_top]
    kem = np.full(len(names), np.nan)
    kem[top] = kemeny(ranks(R[:, top].astype(float))) + 1
    out["Кемени"] = kem
    N = pairwise(R)
    cond = [a for a in range(len(names)) if all(N[a, b] > N[b, a] for b in range(len(names)) if b != a)]
    return out.sort_values("Борда"), (names[cond[0]] if cond else None)


def main():
    cfg = load_config()
    p = cfg["selection"]
    rng = np.random.default_rng(cfg["seed"])
    panel = load_panel(cfg)
    fc = load_forecasts(cfg, p["models"])
    models = [m for m in p["models"] if m in fc]
    hs = cfg["validation"]["horizons"]
    pd.set_option("display.width", 220)

    rows, cells = [], []
    for h in hs:
        L, tid, cat = losses(panel, fc, models, h)
        pv, d = mcs(L, tid, p["alpha"], p["boot"], rng)
        rows += [{"h": h, "category": "Итого", "model": m, "MAE": d[j], "p_mcs": pv[j]} for j, m in enumerate(models)]
        for c in range(6):
            r = cat == c
            pv_c, d_c = mcs(L[r], tid[r], p["alpha"], p["boot"], rng)
            rows += [{"h": h, "category": CATEGORIES[c], "model": m, "MAE": d_c[j], "p_mcs": pv_c[j]}
                     for j, m in enumerate(models)]
            cells.append(d_c)
        print(f"h = {h}: пар {len(L)}, МО {len(np.unique(tid))}", flush=True)
    t = pd.DataFrame(rows)
    t["in_mcs"] = t.p_mcs >= p["alpha"]
    t.to_csv(out_path(cfg, "selection_mcs.csv"), index=False, encoding="utf-8")
    tot = t[t.category == "Итого"].pivot(index="model", columns="h", values="p_mcs")
    print(f"\nMCS, p-значения (в наборе при p >= {p['alpha']}):")
    print(tot.loc[tot.max(1).sort_values(ascending=False).index].round(3).to_string())
    print("\nсколько клеток «горизонт x категория» модель входит в набор лучших:")
    print(t[t.category != "Итого"].groupby("model").in_mcs.sum().sort_values(ascending=False).to_string())

    R = ranks(np.array(cells))
    agg, cond = aggregate(R, np.array(models), p["kemeny_top"])
    agg.to_csv(out_path(cfg, "selection_rank_forecast.csv"), index=False, encoding="utf-8")
    print(f"\nПрогноз: {len(cells)} избирателей, победитель Кондорсе: {cond}")
    print(agg.head(10).to_string(index=False))

    raw = pd.read_csv(out_path(cfg, "detection_synthetic.csv"))
    for name, keep in (("все срезы", raw.group != "все"), ("размер сдвига", raw.group.str.startswith("|delta|"))):
        det = raw[keep].groupby(["group", "alpha", "detector"]).recall.mean().unstack("detector")
        agg_d, cond_d = aggregate(ranks(det.to_numpy(), higher_better=True), np.array(det.columns), p["kemeny_top"])
        agg_d.insert(0, "voters", name)
        agg_d.to_csv(out_path(cfg, f"selection_rank_detect{'' if name == 'все срезы' else '_delta'}.csv"),
                     index=False, encoding="utf-8")
        print(f"\nДетекторы, {name}: {len(det)} избирателей, победитель Кондорсе: {cond_d}")
        print(agg_d.to_string(index=False))


if __name__ == "__main__":
    main()
