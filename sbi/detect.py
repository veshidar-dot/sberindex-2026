"""Детекторы структурных сдвигов. Каждый детектор работает онлайн: оценка S[i, t] считается
только по данным до месяца t включительно; тревога - S[i, t] > порог.

Вход - матрица Z (N, T): логи без общей компоненты категории, нормированные на шум ряда
(sqrt(k_i)), так что у всех рядов шум порядка 1.

  cusum  - двусторонний CUSUM Пейджа на отклонениях от среднего за прошлые месяцы
  pelt   - PELT (ruptures, l2) на данных до t: наибольший штраф из сетки, при котором
           найдена точка разладки в последних w месяцах
  bocpd  - байесовская онлайн-детекция (Adams, MacKay 2007), гауссовы данные с неизвестным
           средним; оценка = P(длина серии <= w-1 | данные до t)
  ssm_lr - LR-тест ступеньки в уровне в модели mu + AR(1) (дополненный фильтр Калмана,
           de Jong 1991), максимум по моменту ступеньки в последних w месяцах
"""
import numpy as np
from scipy.special import logsumexp

from .models.panel_ssm import kalman


def cusum(Z, k=0.5, warm=8):
    """CUSUM Пейджа на остатках AR(1) относительно опорного уровня (контрольная карта для
    автокоррелированных данных): e_t = z_t - rho z_{t-1}, ряды центрированы по первым warm
    месяцам; rho общий (МНК по опорному периоду), остатки нормированы на их sd в опорном периоде.
    При сдвиге уровня на d остаток смещается на d(1 - rho) и сумма накапливается."""
    N, T = Z.shape
    a, b = Z[:, 1:warm].ravel(), Z[:, :warm - 1].ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    rho = (a[ok] @ b[ok]) / (b[ok] @ b[ok])
    E = np.full_like(Z, np.nan)
    E[:, 1:] = Z[:, 1:] - rho * Z[:, :-1]
    sd = np.nanstd(E[:, 1:warm], axis=1)
    E = E / np.where(sd > 0, sd, np.nanmedian(sd))[:, None]
    S = np.zeros((N, T))
    sp = np.zeros(N)
    sm = np.zeros(N)
    for t in range(warm, T):
        x = E[:, t]
        ok = np.isfinite(x)
        sp[ok] = np.maximum(0, sp[ok] + x[ok] - k)
        sm[ok] = np.maximum(0, sm[ok] - x[ok] - k)
        S[:, t] = np.maximum(sp, sm)
    return S


def pelt(Z, w=3, warm=6, pens=(1, 2, 3, 4, 6, 8, 11, 15, 20, 30, 45, 70, 100, 150, 220, 330, 500)):
    import ruptures as rpt
    N, T = Z.shape
    S = np.zeros((N, T))
    for i in range(N):
        for t in range(warm, T):
            y = Z[i, :t + 1]
            y = y[np.isfinite(y)]
            if len(y) < warm:
                continue
            algo = rpt.Pelt(model="l2", min_size=1, jump=1).fit(y.reshape(-1, 1))
            best = 0.0
            for p in pens:                      # штраф в единицах дисперсии шума (=1)
                bk = algo.predict(pen=p)[:-1]
                if any(b > len(y) - 1 - w for b in bk):
                    best = p
                else:
                    break
            S[i, t] = best
    return S


def bocpd(Z, hazard=1 / 24, prior_var=9.0, w=3, hazard_boost=None):
    """Гауссова модель: x_t ~ N(m, 1), m ~ N(0, prior_var) в начале каждого сегмента
    (ряды центрированы по опорному периоду). hazard_boost (N, T) - множитель вероятности
    разладки в месяц t (новости: известное событие в МО)."""
    N, T = Z.shape
    S = np.zeros((N, T))
    for i in range(N):
        logR = np.array([0.0])          # log P(длина серии = r | x_1:t), r = 0..
        mean = np.array([0.0])          # апостериорные среднее и дисперсия m при длине r
        var = np.array([prior_var])
        for t in range(T):
            x = Z[i, t]
            if not np.isfinite(x):
                continue
            h = min(hazard * (hazard_boost[i, t] if hazard_boost is not None else 1.0), 0.99)
            pv = var + 1.0
            logpred = -0.5 * (np.log(2 * np.pi * pv) + (x - mean) ** 2 / pv)
            grow = logR + logpred + np.log(1 - h)
            cp = logsumexp(logR + logpred + np.log(h))
            logR = np.r_[cp, grow]
            logR -= logsumexp(logR)
            nv = 1 / (1 / var + 1)
            nm = nv * (mean / var + x)
            mean = np.r_[0.0, nm]
            var = np.r_[prior_var, nv]
            # логит P(длина серии < w): без насыщения у 1, чтобы пороги различались
            S[i, t] = logsumexp(logR[:w]) - (logsumexp(logR[w:]) if len(logR) > w else -50.0)
    return S


def ssm_lr(Z, lam, phi, w=3, warm=6):
    """Для каждого t и ступеньки в tau из (t-w, t]: LR = (sum v_x v_y / F)^2 / sum v_x^2 / F,
    где v_y, v_x - инновации фильтра для данных и для фиктивной ступеньки (одинаковые веса)."""
    N, T = Z.shape
    S = np.zeros((N, T))
    _, _, _, s, Vy, Fy = kalman(Z, lam, phi, keep=True)
    mask = np.isfinite(Z)
    for tau in range(warm, T):
        X = np.where(mask, (np.arange(T) >= tau).astype(float)[None, :], np.nan)
        _, _, _, _, Vx, _ = kalman(X, lam, phi, keep=True)
        num = np.nancumsum(np.nan_to_num(Vx * Vy / Fy), axis=1)
        den = np.nancumsum(np.nan_to_num(Vx ** 2 / Fy), axis=1)
        lr = np.where(den > 1e-9, num ** 2 / np.maximum(den, 1e-9) / s, 0.0)
        for t in range(tau, min(tau + w, T)):
            S[:, t] = np.maximum(S[:, t], lr[:, t])
    return S


DETECTORS = {"cusum": cusum, "pelt": pelt, "bocpd": bocpd, "ssm_lr": ssm_lr}
