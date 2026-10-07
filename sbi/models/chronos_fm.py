"""Фундаментальная модель Chronos-2 (amazon/chronos-2) без дообучения.

chronos2        - каждый ряд МО x категория отдельно, контекст = вся история до точки начала (логи), медиана.
chronos2_cross  - то же, но cross_learning=True: модель видит батч рядов одной категории совместно.
Третий вариант (Chronos-2 для национального ряда внутри панельной модели) - panel_ssm с national_rule=chronos.
"""
import numpy as np

_PIPE = {}


def pipeline(name="amazon/chronos-2", threads=None):
    if name not in _PIPE:
        import torch
        from chronos import Chronos2Pipeline
        if threads:
            torch.set_num_threads(threads)
        _PIPE[name] = Chronos2Pipeline.from_pretrained(name, device_map="cpu")
    return _PIPE[name]


def median_forecast(pipe, contexts, H, cross=False, batch_size=256):
    q, _ = pipe.predict_quantiles(contexts, prediction_length=H, quantile_levels=[0.5],
                                  cross_learning=cross, batch_size=batch_size)
    return np.stack([x[0, :, 0].numpy() for x in q])


def _run(panel, origins, H, params, cross):
    params = params or {}
    pipe = pipeline(params.get("model", "amazon/chronos-2"), params.get("threads"))
    bs = params.get("batch_size", 256)
    F = np.full((len(origins), panel.n, H), np.nan)
    Y = panel.Y
    for k, o in enumerate(origins):
        has = np.isfinite(Y[:, :o + 1]).any(axis=1)
        for c in range(6):
            rows = np.where(has & (panel.cat == c))[0]
            ctx = []
            for i in rows:
                y = Y[i, :o + 1]
                first = np.argmax(np.isfinite(y))
                ctx.append(y[first:].astype(np.float32))
            F[k, rows] = np.exp(median_forecast(pipe, ctx, H, cross, bs))
        print(f"  chronos origin {panel.months[o]} done", flush=True)
    return F


def chronos2(panel, origins, H, params=None):
    return _run(panel, origins, H, params, cross=False)


def chronos2_cross(panel, origins, H, params=None):
    return _run(panel, origins, H, params, cross=True)
