import numpy as np
import matplotlib.pyplot as plt


from scipy.io import wavfile

WAV_FILE = "engine.wav"
START_S = 600.0
LENGTH_S = 10.0

fs, x = wavfile.read(WAV_FILE)

start = int(START_S * fs)
stop = int((START_S + LENGTH_S) * fs)
x = x[start:stop]

print("Clip sample =", len(x))
print("Clip duration = ", len(x)/ fs)
print(fs )

x = x.astype(float) / 32768.0

t = np.arange(len(x)) / fs

print("Max value = ", x.max())
print("Min value = ", x.min())

plt.plot(t, x)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.xlim(0, 0.01)
plt.show()

from scipy.signal import spectrogram

WINDOW = 16384
OVERLAP = 0.75

f, t_spec, Sxx = spectrogram(
    x,
    fs=fs,
    window="hann",
    nperseg=WINDOW, 
    noverlap=int(WINDOW * OVERLAP),
)

print("f shape = ", f.shape)
print("t_spec shape = ", t_spec.shape)
print("Sxx shape = ", Sxx.shape)
print("Frequency resolution =", fs/WINDOW)
print("First 5 frequencies = ", f[:5])
print("First 5 time =", t_spec[:5])

Sxx_db = 10 * np.log10(Sxx+ 1e-12)
plt.figure(figsize=(10, 5))
plt.pcolormesh( t_spec, f, Sxx_db, shading="auto", cmap="magma", vmin=Sxx_db.max() - 60,vmax = Sxx_db.max(),)

plt.ylim(0,2000)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Spectrogram")
plt.show()

loudest_freqs = []
for j in range (Sxx.shape[1]):
    column = Sxx[:,j]
    row = np.argmax(column)
    loudest_freqs.append(f[row])

loudest_freqs = np.array(loudest_freqs)
rpm_naive = loudest_freqs * 20   # 六缸发动机

print("First 5 loudest freqs =", loudest_freqs[:5])
print("First 5 naive RPM =", rpm_naive[:5])

plt.figure(figsize=(10,4))
plt.plot(t_spec, rpm_naive)
plt.xlabel("Time (s)")
plt.ylabel("Naive RPM estimate")
plt.show()


F_MIN = 40.0
F_MAX = 125.0


band = (f>= F_MIN) & (f<= F_MAX)

f_band = f[band]
Sxx_band = Sxx[band,:]

loudest_in_band = f_band[np.argmax(Sxx_band, axis = 0)]
rpm_band = loudest_in_band * 20

plt.figure(figsize = (10,4))
plt.plot(t_spec, rpm_band)
plt.xlabel("Time (s)")
plt.ylabel("RPM estimate (band-pass)")
plt.show()


candidates = np.arange(40.0, 125.0 + 0.001, 0.25)
HARMONICS = np.arange(2, 14)

score = np.zeros((len(candidates), Sxx_db.shape[1]))

for j in range(Sxx_db.shape[1]):
    column = Sxx_db[:, j]
    on_lines = np.zeros(len(candidates))
    between = np.zeros(len(candidates))
    for k in HARMONICS:
        on_lines += np.interp(k * candidates, f, column)
        between += np.interp((k - 0.5) * candidates, f, column)
    score[:, j] = (on_lines - between) / len(HARMONICS)

best = np.argmax(score, axis=0)
d_est = candidates[best]
rpm_est = d_est * 120

f0_hs = d_est



plt.figure(figsize=(12, 5))
plt.pcolormesh(
    t_spec, f, Sxx_db,
    shading="auto",
    cmap="magma",
    vmin=Sxx_db.max() - 70,
    vmax=Sxx_db.max(),
)
plt.plot(t_spec, f0_hs, ".", color="cyan", markersize=3, label="estimated spacing d")
plt.plot(t_spec, 6 * f0_hs, ".", color="lime", markersize=3, label="6 x d (V6 firing line)")
plt.ylim(0, 1000)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.legend(loc="upper right")
plt.title("Estimate overlaid on spectrogram")
plt.show()


from scipy.ndimage import median_filter

rpm_median = median_filter(rpm_est, size=11, mode="nearest")
bad = np.abs(rpm_est - rpm_median) > 0.05 * rpm_median
print("Outlier frames:", bad.sum(), "of", len(rpm_est))

rpm_clean = rpm_est.copy()
rpm_clean[bad] = np.nan

plt.figure(figsize=(12, 4))
plt.plot(t_spec, rpm_est, ".", color="gray", markersize=4, label="raw estimate")
plt.plot(t_spec, rpm_clean, ".", color="lime", markersize=4, label="after outlier removal")
plt.ylim(0, 15000)
plt.xlabel("Time (s)")
plt.ylabel("Engine speed (RPM)")
plt.title("Engine RPM estimate")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

valid = ~np.isnan(rpm_clean)
t_valid = t_spec[valid]
r_valid = rpm_clean[valid]

ratio = r_valid[1:] / r_valid[:-1]
gap = np.diff(t_valid)

shift_idx = np.where((ratio < 0.91) & (gap < 0.5))[0]

for i in shift_idx:
    print(f"t = {t_valid[i + 1]:.2f} s   {r_valid[i]:.0f} -> {r_valid[i + 1]:.0f} RPM   ratio = {ratio[i]:.3f}")


shift_times = t_valid[shift_idx + 1]

plt.figure(figsize=(12, 4))
plt.plot(t_spec, rpm_clean, ".", color="tab:blue", markersize=4)
for ts in shift_times:
    plt.axvline(ts, color="red", linestyle="--", alpha=0.6)
plt.ylim(0, 15000)
plt.xlabel("Time (s)")
plt.ylabel("Engine speed (RPM)")
plt.title("Estimated engine RPM with detected gear shifts")
plt.grid(alpha=0.3)
plt.savefig("rpm_with_shifts.png", dpi=150, bbox_inches="tight")
plt.show()

out = np.column_stack([t_spec + START_S, rpm_clean])
np.savetxt("rpm_estimate.csv", out, delimiter=",", header="time_s,rpm", comments="", fmt="%.3f")