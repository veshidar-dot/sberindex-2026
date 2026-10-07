"""Центральная модель: панельная модель пространства состояний.

    log y_it = n_ct + z_it,   z_it = mu_it + r_it
    mu_it = mu_i,t-1 + eta_it,      eta ~ N(0, lam * s * k_i)    - постоянный уровень МО (сдвиги)
    r_it  = phi * r_i,t-1 + u_it,   u   ~ N(0, s * k_i)          - временные отклонения, AR(1)

n_ct - общая компонента категории (двусторонние эффекты), её прогноз - sbi.national.
k_i - масштаб шума ряда (робастная дисперсия первых разностей z_i): мелкие МО шумнее.
Параметры (lam, phi) общие для категории, оцениваются ММП по всей панели категории:
фильтр Калмана векторизован по МО, масштаб s концентрирован. Точечный прогноз зависит
только от (lam, phi), поэтому k_i влияет на веса в правдоподобии и на детекцию сдвигов.
"""
import numpy as np
from scipy.optimize import minimize

from ..national import forecast_national, load_external, two_way

DIFFUSE = 1e4


def series_scale(Z, floor=1e-4):
    d = np.diff(Z, axis=1)
    med = np.nanmedian(d, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(d - med), axis=1)
    k = (1.4826 * mad) ** 2
    k = np.where(np.isfinite(k), k, np.nanmedian(k))
    return np.maximum(k, floor)


def kalman(Z, lam, phi, keep=False, qboost=None):
    """Фильтр по всем рядам сразу. Z (N, T) уже нормирован на sqrt(k_i), пропуски - NaN.
    qboost (N, T) - множитель дисперсии шока уровня (новости о событии в МО в месяц t).
    Возвращает -2 logL с концентрированным масштабом, отфильтрованные состояния на конце
    и (keep=True) инновации и их дисперсии для детекции."""
    N, T = Z.shape
    m = np.zeros((N, 2))
    P = np.zeros((N, 2, 2))
    P[:, 0, 0] = DIFFUSE
    P[:, 1, 1] = 1.0 / max(1 - phi ** 2, 1e-3)
    Tm = np.array([[1.0, 0.0], [0.0, phi]])
    Q = np.diag([lam, 1.0])
    seen = np.zeros(N, bool)
    ss, nobs, logdet = 0.0, 0, 0.0
    V = np.full((N, T), np.nan) if keep else None
    Fs = np.full((N, T), np.nan) if keep else None
    for t in range(T):
        if t > 0:
            m = m @ Tm.T
            P = Tm @ P @ Tm.T + Q
            if qboost is not None:
                P[:, 0, 0] += lam * (qboost[:, t] - 1)
        y = Z[:, t]
        ok = np.isfinite(y)
        v = y - m[:, 0] - m[:, 1]
        F = P[:, 0, 0] + P[:, 1, 1] + 2 * P[:, 0, 1]
        K = (P[:, :, 0] + P[:, :, 1]) / F[:, None]
        count = ok & seen                      # первое наблюдение ряда - диффузная инициализация
        ss += np.sum(v[count] ** 2 / F[count])
        logdet += np.sum(np.log(F[count]))
        nobs += count.sum()
        if keep:
            V[count, t] = v[count]
            Fs[count, t] = F[count]
        m[ok] += K[ok] * v[ok, None]
        P[ok] -= K[ok][:, :, None] * (P[ok, 0, :] + P[ok, 1, :])[:, None, :]
        seen |= ok
    s = ss / max(nobs, 1)
    m2ll = nobs * np.log(s) + logdet
    return m2ll, m, P, s, V, Fs


def fit(Z, qboost=None):
    def obj(th):
        lam, phi = np.exp(th[0]), np.tanh(th[1])
        return kalman(Z, lam, phi, qboost=qboost)[0]
    best = None
    for l0 in (-4.0, -1.0):
        r = minimize(obj, x0=[l0, 0.7], method="Nelder-Mead",
                     options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 200})
        if best is None or r.fun < best.fun:
            best = r
    return float(np.exp(best.x[0])), float(np.tanh(best.x[1])), float(best.fun)


def panel_ssm(panel, origins, H, params=None):
    from ..config import load_config
    params = {} if params is None else params
    cfg = load_config()
    ext = load_external(cfg)
    rule = params.get("national_rule", "combo")
    news_factor = params.get("news_factor", 1.0)
    events = None
    if news_factor > 1:
        from ..news import boost_matrix, load_events
        events = load_events(cfg)
    F = np.full((len(origins), panel.n, H), np.nan)
    hs = np.arange(1, H + 1)
    log = []
    for c in range(6):
        rows = np.where(panel.cat == c)[0]
        for k, o in enumerate(origins):
            Yc = panel.Y[rows, :o + 1]
            a, n = two_way(Yc)
            Z = Yc - n
            kap = series_scale(Z - a[:, None])
            Zs = Z / np.sqrt(kap)[:, None]
            qb = None
            if events is not None:
                qb = boost_matrix(panel, events, news_factor, known_at=panel.months[o])[rows, :o + 1]
            lam, phi, m2ll = fit(Zs, qb)
            _, m, _, _, _, _ = kalman(Zs, lam, phi, qboost=qb)
            mu, r = m[:, 0] * np.sqrt(kap), m[:, 1] * np.sqrt(kap)
            nf = forecast_national(n, panel.months, o, c, ext, H, {"chronos": rule == "chronos", "ext_map": params.get("ext_map", "fixed")})[rule]
            F[k, rows] = np.exp(nf[None, :] + mu[:, None] + phi ** hs[None, :] * r[:, None])
            log.append((c, panel.months[o], lam, phi))
    params["_fitted"] = log
    return F
