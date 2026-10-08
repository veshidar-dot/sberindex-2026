"""Проверки «без заглядывания вперёд»: прогноз из точки начала o не должен меняться, если испортить
всё, что известно только после o (значения панели, национальный ряд, новости, события).

    python -m pytest -q tests
"""
import copy

import numpy as np
import pandas as pd
import pytest

from sbi.config import load_config
from sbi.data import load_panel
from sbi.models.baselines import naive, seasonal_naive
from sbi.models.panel_seasonal import panel_seasonal
from sbi.models.panel_ssm import panel_ssm
from sbi.national import forecast_national, load_external, two_way
from sbi.news import boost_matrix, load_events

ORIGIN = "2024-03"
H = 6


@pytest.fixture(scope="module")
def small_panel():
    cfg = load_config()
    p = load_panel(cfg)
    rng = np.random.default_rng(0)
    rows = np.sort(np.concatenate([rng.choice(np.where(p.cat == c)[0], 60, replace=False) for c in range(6)]))
    q = copy.copy(p)
    q.keys = p.keys.iloc[rows].reset_index(drop=True)
    q.V = p.V[rows].copy()
    return q


def spoil_future(panel, o):
    """Копия панели, где все значения после месяца o заменены мусором."""
    q = copy.copy(panel)
    q.V = panel.V.copy()
    q.V[:, o + 1:] = np.random.default_rng(1).uniform(1, 1e6, q.V[:, o + 1:].shape)
    return q


@pytest.mark.parametrize("model", [naive, seasonal_naive, panel_seasonal, panel_ssm])
def test_model_ignores_future_panel(small_panel, model):
    o = small_panel.month_index(ORIGIN)
    a = model(small_panel, [o], H, {})
    b = model(spoil_future(small_panel, o), [o], H, {})
    np.testing.assert_allclose(a, b, rtol=1e-9, equal_nan=True)


@pytest.mark.parametrize("model", [naive, panel_seasonal, panel_ssm])
def test_control_past_changes_forecast(small_panel, model):
    """Контроль чувствительности: порча последнего известного месяца обязана изменить прогноз."""
    o = small_panel.month_index(ORIGIN)
    q = copy.copy(small_panel)
    q.V = small_panel.V.copy()
    q.V[:, o] *= 1.5
    a = model(small_panel, [o], H, {})
    b = model(q, [o], H, {})
    assert np.nanmax(np.abs(a - b)) > 1.0


def test_national_rules_ignore_future_external(small_panel):
    cfg = load_config()
    ext = load_external(cfg)
    o = small_panel.month_index(ORIGIN)
    _, n = two_way(small_panel.Y[small_panel.cat == 0, :o + 1])
    spoiled = ext.copy()
    spoiled.loc[spoiled.index > ORIGIN] = 99.0
    a = forecast_national(n, small_panel.months, o, 0, ext, H)
    b = forecast_national(n, small_panel.months, o, 0, spoiled, H)
    for rule in a:
        np.testing.assert_allclose(a[rule], b[rule], rtol=1e-9, err_msg=rule)


def test_events_respect_announce_date(small_panel):
    """Событие, о котором объявили после точки начала, не должно влиять на матрицу новостей."""
    cfg = load_config()
    ev = load_events(cfg).copy()
    o_month = "2024-03"
    late = ev.event_month >= "2024-04"
    B = boost_matrix(small_panel, ev, 25, known_at=o_month)
    ev2 = ev.copy()
    ev2.loc[late, "announce_date"] = pd.Timestamp("2030-01-01")
    B2 = boost_matrix(small_panel, ev2, 25, known_at=o_month)
    np.testing.assert_array_equal(B, B2)


def test_gdelt_surprise_uses_only_past(small_panel):
    """Неожиданность новостей в месяц t зависит только от месяцев <= t."""
    from sbi import gdelt
    S, _ = gdelt.surprise_matrix(small_panel, "hazard")
    orig = gdelt.load_gdelt

    def spoiled(level="mo"):
        d = orig(level)
        d.loc[d.month > ORIGIN, ["hazard", "docs", "neg_docs"]] = 10 ** 6
        return d

    gdelt.load_gdelt = spoiled
    try:
        S2, _ = gdelt.surprise_matrix(small_panel, "hazard")
    finally:
        gdelt.load_gdelt = orig
    o = small_panel.month_index(ORIGIN)
    np.testing.assert_allclose(S[:, :o + 1], S2[:, :o + 1], equal_nan=True)
