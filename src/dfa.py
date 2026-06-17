import numpy as np


def make_scales(n, min_scale=16, max_scale=None, count=24):
    if max_scale is None:
        max_scale = max(min_scale + 1, n // 5)
    scales = np.unique(np.logspace(np.log10(min_scale), np.log10(max_scale), count).astype(int))
    return scales[(scales >= min_scale) & (scales < n // 2)]


def _segment_variances(profile, scale, order):
    n = profile.size
    segment_count = n // scale
    if segment_count < 2:
        return np.array([])

    variances = []
    x = np.arange(scale)
    for start in list(range(0, segment_count * scale, scale)) + list(
        range(n - segment_count * scale, n, scale)
    ):
        segment = profile[start : start + scale]
        if segment.size != scale:
            continue
        coeffs = np.polyfit(x, segment, order)
        trend = np.polyval(coeffs, x)
        variances.append(np.mean((segment - trend) ** 2))
    return np.asarray(variances)


def dfa(series, scales=None, order=1):
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    profile = np.cumsum(values - values.mean())
    if scales is None:
        scales = make_scales(profile.size)

    used_scales = []
    fluctuations = []
    for scale in scales:
        variances = _segment_variances(profile, int(scale), order)
        if variances.size == 0:
            continue
        used_scales.append(scale)
        fluctuations.append(np.sqrt(np.mean(variances)))

    used_scales = np.asarray(used_scales, dtype=float)
    fluctuations = np.asarray(fluctuations, dtype=float)
    log_s = np.log(used_scales)
    log_f = np.log(fluctuations)
    slope, intercept = np.polyfit(log_s, log_f, 1)
    fitted = intercept + slope * log_s
    ss_res = np.sum((log_f - fitted) ** 2)
    ss_tot = np.sum((log_f - log_f.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "scales": used_scales,
        "fluctuations": fluctuations,
        "alpha": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
    }


def mfdfa(series, q_values, scales=None, order=1):
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    profile = np.cumsum(values - values.mean())
    if scales is None:
        scales = make_scales(profile.size)

    fluctuation_matrix = np.full((len(q_values), len(scales)), np.nan)
    used_scales = np.asarray(scales, dtype=float)

    for scale_index, scale in enumerate(scales):
        variances = _segment_variances(profile, int(scale), order)
        variances = variances[variances > 0]
        if variances.size == 0:
            continue
        rms = np.sqrt(variances)
        for q_index, q in enumerate(q_values):
            if np.isclose(q, 0.0):
                fluctuation_matrix[q_index, scale_index] = np.exp(np.mean(np.log(rms)))
            else:
                fluctuation_matrix[q_index, scale_index] = (
                    np.mean(rms ** q) ** (1.0 / q)
                )

    hq = []
    for row in fluctuation_matrix:
        mask = np.isfinite(row) & (row > 0)
        slope, _ = np.polyfit(np.log(used_scales[mask]), np.log(row[mask]), 1)
        hq.append(slope)
    hq = np.asarray(hq)

    tau = q_values * hq - 1.0
    alpha = np.gradient(tau, q_values)
    f_alpha = q_values * alpha - tau
    return {
        "q": q_values,
        "scales": used_scales,
        "fluctuations": fluctuation_matrix,
        "hq": hq,
        "tau": tau,
        "alpha": alpha,
        "f_alpha": f_alpha,
        "spectrum_width": float(np.nanmax(alpha) - np.nanmin(alpha)),
    }
