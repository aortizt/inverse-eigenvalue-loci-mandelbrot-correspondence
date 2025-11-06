import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- Load ordered boundary data ---
C = np.loadtxt("/home/Merlin/Desktop/out_clean/construct_aligned.csv", delimiter=",")
M = np.loadtxt("/home/Merlin/Desktop/out_clean/mandel_boundary_sample.csv", delimiter=",")

def order_points(points):
    centroid = points.mean(axis=0)
    angles = np.arctan2(points[:,1]-centroid[1], points[:,0]-centroid[0])
    idx = np.argsort(angles)
    return points[idx]

C_ord = order_points(C)
M_ord = order_points(M)

# --- Convert to complex signals ---
zC = C_ord[:,0] + 1j*C_ord[:,1]
zM = M_ord[:,0] + 1j*M_ord[:,1]

# --- FFT and amplitudes ---
fftC = np.fft.fft(zC - np.mean(zC))
fftM = np.fft.fft(zM - np.mean(zM))

freqsC = np.fft.fftfreq(len(fftC))
freqsM = np.fft.fftfreq(len(fftM))

maskC = freqsC > 0
maskM = freqsM > 0

freqC = freqsC[maskC]
freqM = freqsM[maskM]

ampC = np.abs(fftC[maskC])
ampM = np.abs(fftM[maskM])

# --- Define frequency ranges for fitting ---
ranges = [(1e-4,1e-3),(1e-3,1e-2),(1e-2,1e-1),(1e-1,0.5)]

results = []

plt.figure(figsize=(10,6))
plt.loglog(freqC, ampC/np.max(ampC), label="Construct", alpha=0.7)
plt.loglog(freqM, ampM/np.max(ampM), label="Mandelbrot", alpha=0.7)

for (fmin,fmax) in ranges:
    for label, freq, amp in [("Construct", freqC, ampC),("Mandelbrot", freqM, ampM)]:
        mask = (freq>=fmin)&(freq<=fmax)
        if np.sum(mask)<5: 
            continue
        X = np.log10(freq[mask]).reshape(-1,1)
        y = np.log10(amp[mask]/np.max(amp))
        reg = LinearRegression().fit(X,y)
        slope = reg.coef_[0]
        R2 = reg.score(X,y)
        results.append((label,fmin,fmax,slope,R2))
        # plot fit line
        xx = np.linspace(np.log10(fmin),np.log10(fmax),100)
        yy = reg.predict(xx.reshape(-1,1))
        plt.plot(10**xx,10**yy, '--', label=f"{label} fit {fmin}-{fmax} Hz (s={slope:.2f})")

plt.xlabel("Frequency")
plt.ylabel("Normalized amplitude")
plt.title("Spectral decay exponent fit")
plt.legend(fontsize=8)
plt.show()

# --- Save results ---
with open("spectral_slope_results.txt","w") as f:
    f.write("Label, fmin, fmax, slope, R2\n")
    for r in results:
        f.write(",".join(map(str,r))+"\n")

print("Results saved to spectral_slope_results.txt")
