import numpy as np
import matplotlib.pyplot as plt

# --- Load data ---
C = np.loadtxt("/home/Merlin/Desktop/out_clean/construct_aligned.csv", delimiter=",")
M = np.loadtxt("/home/Merlin/Desktop/out_clean/mandel_boundary_sample.csv", delimiter=",")

# --- Helper: order points by angle around centroid ---
def order_points(points):
    centroid = points.mean(axis=0)
    angles = np.arctan2(points[:,1] - centroid[1], points[:,0] - centroid[0])
    idx = np.argsort(angles)
    return points[idx]

# Order Construct and Mandelbrot points
C_ord = order_points(C)
M_ord = order_points(M)

# --- Convert to complex signal ---
zC = C_ord[:,0] + 1j*C_ord[:,1]
zM = M_ord[:,0] + 1j*M_ord[:,1]

# --- FFT ---
fftC = np.fft.fft(zC - np.mean(zC))   # remove centroid shift
fftM = np.fft.fft(zM - np.mean(zM))

freqsC = np.fft.fftfreq(len(fftC))
freqsM = np.fft.fftfreq(len(fftM))

# --- Spectra (magnitude) ---
ampC = np.abs(fftC)
ampM = np.abs(fftM)

# Normalize for comparison
ampC /= np.max(ampC)
ampM /= np.max(ampM)

# --- Plot spectra ---
plt.figure(figsize=(10,6))
plt.plot(np.abs(freqsC), ampC, label="Construct spectrum", alpha=0.7)
plt.plot(np.abs(freqsM), ampM, label="Mandelbrot spectrum", alpha=0.7)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Frequency (log scale)")
plt.ylabel("Normalized amplitude (log scale)")
plt.title("Fourier spectral comparison: Construct vs Mandelbrot")
plt.legend()
plt.show()

# --- Print first few modes ---
n_modes = 10
print("First {} Fourier modes (normalized amplitudes):".format(n_modes))
print("Mode | Construct | Mandelbrot")
for k in range(1, n_modes+1):
    print(f"{k:4d} | {ampC[k]:.4f} | {ampM[k]:.4f}")

# --- Inverse FFT reconstruction ---
def reconstruct(fft_coeffs, n_modes):
    coeffs = np.zeros_like(fft_coeffs, dtype=complex)
    coeffs[:n_modes] = fft_coeffs[:n_modes]
    coeffs[-n_modes+1:] = fft_coeffs[-n_modes+1:]
    return np.fft.ifft(coeffs)

modes_to_show = [5, 10, 30, 100]

plt.figure(figsize=(12,6))
for i, nm in enumerate(modes_to_show, 1):
    recC = reconstruct(fftC, nm)
    recM = reconstruct(fftM, nm)
    plt.subplot(2, len(modes_to_show)//2, i)
    plt.plot(recC.real, recC.imag, label=f"Construct {nm} modes", alpha=0.7)
    plt.plot(recM.real, recM.imag, label=f"Mandelbrot {nm} modes", alpha=0.7)
    plt.axis("equal")
    plt.legend(fontsize=8)
    plt.title(f"Reconstruction with {nm} modes")

plt.tight_layout()
plt.show()