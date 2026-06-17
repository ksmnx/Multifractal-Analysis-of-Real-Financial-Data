import numpy as np


def fractional_gaussian_noise(n, hurst, seed=None):
    """Approximate fGn sample. This is enough for a benchmark, not calibration."""
    if not 0 < hurst < 1:
        raise ValueError("hurst must be between 0 and 1")

    rng = np.random.default_rng(seed)
    k = np.arange(0, n)
    gamma = 0.5 * (
        np.abs(k - 1) ** (2 * hurst)
        - 2 * np.abs(k) ** (2 * hurst)
        + np.abs(k + 1) ** (2 * hurst)
    )
    circulant = np.concatenate([gamma, [0.0], gamma[1:][::-1]])
    eigenvalues = np.real(np.fft.fft(circulant))
    eigenvalues[eigenvalues < 0] = 0.0

    gaussian = rng.normal(size=2 * n) + 1j * rng.normal(size=2 * n)
    sample = np.fft.ifft(np.sqrt(eigenvalues) * gaussian).real[:n]
    sample = (sample - sample.mean()) / sample.std(ddof=1)
    return sample


def fractional_brownian_motion(n, hurst, seed=None):
    return np.cumsum(fractional_gaussian_noise(n, hurst, seed=seed))
