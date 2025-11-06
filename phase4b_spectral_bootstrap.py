
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

# === Helper functions ===
def compute_spectrum(points, n_freq=1000):
    """Compute normalized Fourier spectrum of 2D points."""
    signal = points[:, 0] + 1j * points[:, 1]
    spectrum = np.abs(fft(signal))**2
    freqs = np.fft.fftfreq(len(signal))
    mask = freqs > 0
    return freqs[mask], spectrum[mask] / np.max(spectrum[mask])

def fit_slope(freqs, spectrum, fmin, fmax, n_bootstrap=200):
    """Fit slope of log-log spectrum between fmin and fmax, with bootstrapped CIs."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    X = np.log10(freqs[mask]).reshape(-1, 1)
    y = np.log10(spectrum[mask])

    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]
    r2 = model.score(X, y)

    # Bootstrap slopes
    slopes = []
    for _ in range(n_bootstrap):
        X_res, y_res = resample(X, y)
        m = LinearRegression().fit(X_res, y_res)
        slopes.append(m.coef_[0])
    slopes = np.array(slopes)
    ci_lower, ci_upper = np.percentile(slopes, [2.5, 97.5])

    return slope, r2, (ci_lower, ci_upper)

# === Load data ===
construct = pd.read_csv("construct_points.csv").values
mandel = pd.read_csv("mandel_boundary_sample.csv").values

# === Compute spectra ===
freq_c, spec_c = compute_spectrum(construct)
freq_m, spec_m = compute_spectrum(mandel)

# === Define ranges for slope fitting ===
ranges = [(1e-3, 1e-2), (1e-2, 1e-1)]

print("=== Spectral Slope Analysis with Bootstrap CIs ===")
for (fmin, fmax) in ranges:
    slope_c, r2_c, ci_c = fit_slope(freq_c, spec_c, fmin, fmax)
    slope_m, r2_m, ci_m = fit_slope(freq_m, spec_m, fmin, fmax)
    print(f"Range {fmin:.0e}–{fmax:.0e}:")
    print(f"  Construct slope = {slope_c:.3f} (95% CI {ci_c[0]:.3f}, {ci_c[1]:.3f}), R²={r2_c:.3f}")
    print(f"  Mandelbrot slope = {slope_m:.3f} (95% CI {ci_m[0]:.3f}, {ci_m[1]:.3f}), R²={r2_m:.3f}")

# === Plot ===
plt.figure(figsize=(8,6))
plt.loglog(freq_c, spec_c, label="Construct")
plt.loglog(freq_m, spec_m, label="Mandelbrot")
plt.xlabel("Frequency (log scale)")
plt.ylabel("Normalized Power Spectrum (log scale)")
plt.title("Fourier Spectral Comparison (Phase 4b with Bootstrap)")
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.savefig("phase4b_spectral_bootstrap.png", dpi=300)
plt.show()
