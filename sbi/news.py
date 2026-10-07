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
