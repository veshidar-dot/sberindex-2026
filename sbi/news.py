"""Новости: каталог событий (ref/events.csv), привязанный к МО и месяцу.

Согласование с данными: territory_id из справочника СберИндекса, месяц события в формате
панели ('YYYY-MM'). Событие можно использовать в точке начала o, только если о нём было
известно к концу месяца o (announce_date <= последний день месяца o) - без заглядывания вперёд.

Матрица boost[i, t] - множитель «вероятности сдвига» для ряда i в месяц t:
  - в panel_ssm умножает дисперсию шока уровня (фильтр быстрее принимает новый уровень);
  - в BOCPD умножает вероятность разладки (hazard).
"""
import numpy as np
import pandas as pd

from .config import data_path


def load_events(cfg):
    e = pd.read_csv(data_path(cfg, "events"), dtype={"territory_ids": str})
    e["ids"] = e.territory_ids.str.split(";").apply(lambda x: [int(v) for v in x])
    e["announce_date"] = pd.to_datetime(e.announce_date)
    return e


def merge_catalogs(ours, maloyan):
    """Наш каталог + события реестра maloyan с привязкой к конкретному МО (паводки, пожары).
    Одно событие = одно МО; пара (МО, месяц), уже описанная в нашем каталоге, не дублируется."""
    o = pd.read_csv(ours, dtype={"territory_ids": str})
    m = pd.read_csv(maloyan)
    m = m[(m.scope == "mo") & m.territory_id.notna() & (m.verified == 1)].copy()
    have = {(int(t), r.event_month) for r in o.itertuples() for t in r.territory_ids.split(";")}
    rows = []
    for r in m.itertuples():
        tid, month = int(r.territory_id), str(r.effective_date)[:7]
        if (tid, month) in have:
            continue
        rows.append({"event_id": f"M{r.event_id}", "title": f"{r.event_type}: {r.mo_name}", "type": r.event_type,
                     "region": r.region_name, "territory_ids": str(tid), "event_month": month,
                     "announce_date": r.announce_date, "expected_sign": r.expected_sign,
                     "source": f"{r.source_url} (реестр maloyan, MIT)"})
    return pd.concat([o, pd.DataFrame(rows)], ignore_index=True)


def boost_matrix(panel, events, factor, known_at=None, spread=(0, 1)):
    """factor - во сколько раз растёт вероятность/дисперсия сдвига в месяц события
    (и следующий, spread). known_at - месяц 'YYYY-MM' точки начала: только известные события."""
    B = np.ones((panel.n, panel.T))
    tid = panel.keys["territory_id"].to_numpy()
    cutoff = pd.Period(known_at, "M").end_time if known_at else None
    for _, ev in events.iterrows():
        if cutoff is not None and ev.announce_date > cutoff:
            continue
        if ev.event_month not in panel.months:
            continue
        t0 = panel.month_index(ev.event_month)
        rows = np.isin(tid, ev.ids)
        for d in spread:
            if t0 + d < panel.T:
                B[rows, t0 + d] = factor
    return B


if __name__ == "__main__":
    # python -m sbi.news - собрать объединённый каталог ref/events_merged.csv
    from .config import resolve
    merged = merge_catalogs(resolve("ref/events.csv"), resolve("ref/news/event_registry_maloyan.csv"))
    merged.to_csv(resolve("ref/events_merged.csv"), index=False, encoding="utf-8")
    ids = {t for s in merged.territory_ids for t in s.split(";")}
    print(f"событий {len(merged)}, МО {len(ids)}")
