"""Фундаментальные модели внутри разложения и TimesFM 2.5.

  timesfm          - TimesFM 2.5 (google/timesfm-2.5-200m-pytorch) по каждому ряду отдельно (логи), без дообучения
  timesfm_local    - TimesFM 2.5 прогнозирует локальную компоненту z = log y - n_ct, общая траектория n_ct -
                     из sbi.national (как в panel_ssm); прогноз = exp(n_hat + z_hat)
  chronos2_local   - то же с Chronos-2

Локальная компонента очищена от общей сезонности и роста категории, поэтому короткий контекст
(12-23 мес.) не заставляет модель принимать декабрьский пик за тренд.
"""
import numpy as np

from ..config import load_config
from ..national import forecast_national, load_external, two_way

_TFM = {}


def timesfm_model(name="google/timesfm-2.5-200m-pytorch", threads=None, batch=256):
    if name not in _TFM:
        import timesfm
        import torch
        if threads:
            torch.set_num_threads(threads)
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(name)
        m.compile(timesfm.ForecastConfig(max_context=32, max_horizon=16, normalize_inputs=True,
                                         per_core_batch_size=batch, infer_is_positive=False))
        _TFM[name] = m
    return _TFM[name]


_TIREX = {}


def _predict(kind, contexts, H, params):
    if kind == "tirex":
        import tirex
        if "m" not in _TIREX:
            _TIREX["m"] = tirex.load_model(params.get("model", "NX-AI/TiRex"), device="cpu", backend="torch")
        q, _ = _TIREX["m"].forecast(context=contexts, prediction_length=H, output_type="numpy",
                                     batch_size=params.get("batch_size", 512))
        return np.asarray(q)[:, :H, 4]          # квантили 0.1..0.9, медиана - пятая
    if kind == "timesfm":
        m = timesfm_model(params.get("model", "google/timesfm-2.5-200m-pytorch"), params.get("threads"),
                          params.get("batch_size", 256))
        point, _ = m.forecast(horizon=H, inputs=contexts)
        return np.asarray(point)[:, :H]
    from .chronos_fm import median_forecast, pipeline
    return median_forecast(pipeline(params.get("model", "amazon/chronos-2"), params.get("threads")),
                           contexts, H, False, params.get("batch_size", 256))


def _trim(y):
    first = np.argmax(np.isfinite(y))
    y = y[first:].astype(np.float32)
    # пропуски внутри ряда - линейная интерполяция (TimesFM не принимает NaN)
    if np.isnan(y).any():
        idx = np.arange(len(y))
        ok = np.isfinite(y)
        y = np.interp(idx, idx[ok], y[ok]).astype(np.float32)
    return y


def _run(panel, origins, H, params, kind, local):
    """sample_per_category - считать только на подвыборке рядов (для медленных моделей); остальные ряды - NaN."""
    params = params or {}
    ext = load_external(load_config()) if local else None
    F = np.full((len(origins), panel.n, H), np.nan)
    keep = np.ones(panel.n, bool)
    if params.get("sample_per_category"):
        rng = np.random.default_rng(load_config()["seed"])
        keep[:] = False
        for c in range(6):
            cand = np.where(panel.cat == c)[0]
            keep[rng.choice(cand, min(params["sample_per_category"], len(cand)), replace=False)] = True
    for k, o in enumerate(origins):
        for c in range(6):
            rows = np.where(panel.cat == c)[0]
            Y = panel.Y[rows, :o + 1]
            has = np.isfinite(Y).any(1) & keep[rows]
            if local:
                _, n = two_way(Y)
                Z = Y - n
                nf = forecast_national(n, panel.months, o, c, ext, H, {"ext_map": params.get("ext_map", "fixed")})[
                    params.get("national_rule", "combo")]
            else:
                Z, nf = Y, np.zeros(H)
            ctx = [_trim(Z[i]) for i in np.where(has)[0]]
            zf = _predict(kind, ctx, H, params)
            F[k, rows[has]] = np.exp(nf[None, :] + zf)
        print(f"  {kind}{'_local' if local else ''} origin {panel.months[o]} done", flush=True)
    return F


def tirex_local(panel, origins, H, params=None):
    return _run(panel, origins, H, params, "tirex", local=True)


def timesfm(panel, origins, H, params=None):
    return _run(panel, origins, H, params, "timesfm", local=False)


def timesfm_local(panel, origins, H, params=None):
    return _run(panel, origins, H, params, "timesfm", local=True)


def chronos2_local(panel, origins, H, params=None):
    return _run(panel, origins, H, params, "chronos", local=True)


def chronos2_ft_local(panel, origins, H, params=None):
    """Chronos-2, дообученный на панели. Для каждой точки начала o - своя копия модели, дообученная
    (LoRA) на локальных компонентах всех рядов только по данным до o: окна «контекст - следующие
    ft_horizon месяцев» нарезаются внутри известной истории. Затем прогноз локальной компоненты,
    общая траектория - из правил sbi.national, как в chronos2_local."""
    import torch
    from .chronos_fm import median_forecast, pipeline
    from ..config import out_path
    params = dict(params or {})
    cfg = load_config()
    ext = load_external(cfg)
    if params.get("threads"):
        torch.set_num_threads(params["threads"])
    base = pipeline(params.get("model", "amazon/chronos-2"))
    F = np.full((len(origins), panel.n, H), np.nan)
    for k, o in enumerate(origins):
        Z_all, nf_all, rows_all = [], {}, {}
        for c in range(6):
            rows = np.where(panel.cat == c)[0]
            Y = panel.Y[rows, :o + 1]
            has = np.isfinite(Y).any(1)
            _, n = two_way(Y)
            Z = Y - n
            nf_all[c] = forecast_national(n, panel.months, o, c, ext, H,
                                          {"ext_map": params.get("ext_map", "fixed")})[params.get("national_rule", "combo")]
            rows_all[c] = (rows[has], [_trim(Z[i]) for i in np.where(has)[0]])
            Z_all += rows_all[c][1]
        train = [z for z in Z_all if len(z) >= params.get("ft_min_len", 10)]
        tuned = base.fit(train, prediction_length=params.get("ft_horizon", 6), finetune_mode=params.get("ft_mode", "lora"),
                         learning_rate=params.get("ft_lr", 1e-5), num_steps=params.get("ft_steps", 200),
                         batch_size=params.get("ft_batch", 128), min_past=params.get("ft_min_past", 4),
                         output_dir=str(out_path(cfg, "cache", "chronos_ft", panel.months[o])),
                         remove_printer_callback=True, report_to=[], save_strategy="no", seed=cfg["seed"])
        for c in range(6):
            rows, ctx = rows_all[c]
            zf = median_forecast(tuned, ctx, H, False, params.get("batch_size", 256))
            F[k, rows] = np.exp(nf_all[c][None, :] + zf)
        print(f"  chronos2_ft_local origin {panel.months[o]}: дообучение на {len(train)} рядах", flush=True)
        del tuned
    return F
