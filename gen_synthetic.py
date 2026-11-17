"""
gen_synthetic.py —— 生成一段"答案已知"的合成 F1 引擎声

运行后会得到三个文件（保存在脚本所在的文件夹）：
  synthetic_engine.wav   合成的引擎声音（可以拖进 Audacity 看频谱图）
  truth_rpm.csv          每 0.01 秒一行的"真实转速"（标准答案）
  rpm_truth.png          真实转速曲线的图

物理关系（V6 四冲程，每转一圈点火 3 次）：
  基频 f0 = RPM / 20      （例如 12000 转/分 -> 600 Hz）
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# ------------------------------------------------------------------
# 1. 参数（可以随便改，观察结果怎么变）
# ------------------------------------------------------------------
FS = 44100          # 采样率（每秒采样点数）
DURATION = 24.0     # 总时长（秒）
NOISE_LEVEL = 0.3   # 噪声强度，0 = 没有噪声，1 = 噪声和信号一样大
SEED = 0            # 随机数种子，固定后每次生成的噪声都一样

# 谐波强度：{第几倍频: 相对振幅}
# 故意让 2 倍频(1.0) 比基频(0.5) 更强，用来测试"倍频误判"
HARMONICS = {1: 0.5, 2: 1.0, 3: 0.6, 4: 0.3}

# ------------------------------------------------------------------
# 2. 定义"真实转速"曲线：用几个关键点描述，其余部分自动连线
#    每次升挡转速会很快掉下来，然后重新爬升
# ------------------------------------------------------------------
key_times = [0.0, 4.0, 4.15, 8.5, 8.65, 13.5, 13.65, 19.0, 19.15, 24.0]
key_rpms = [6000, 14500, 10800, 14500, 11200, 14500, 11800, 14500, 12200, 14700]

t = np.arange(int(FS * DURATION)) / FS          # 每个采样点对应的时间
rpm = np.interp(t, key_times, key_rpms)         # 每个采样点的转速
f0 = rpm / 20.0                                 # 每个采样点的基频 (Hz)

# ------------------------------------------------------------------
# 3. 生成声音
#    频率在变化，所以不能直接 sin(2*pi*f*t)，
#    要先把频率对时间累加（积分）得到相位，再对相位取 sin。
# ------------------------------------------------------------------
phase = 2 * np.pi * np.cumsum(f0) / FS

signal = np.zeros_like(t)
for k, amplitude in HARMONICS.items():
    signal += amplitude * np.sin(k * phase)

signal /= np.max(np.abs(signal))                # 先把纯信号缩放到最大值 = 1

rng = np.random.default_rng(SEED)
noise = rng.standard_normal(len(t))
noise /= np.max(np.abs(noise))                  # 噪声也缩放到最大值 = 1
signal = signal + NOISE_LEVEL * noise

signal = 0.8 * signal / np.max(np.abs(signal))  # 总体音量留一点余量，防止削波

# ------------------------------------------------------------------
# 4. 保存文件
# ------------------------------------------------------------------
out_dir = Path(__file__).resolve().parent

wav_path = out_dir / "synthetic_engine.wav"
wavfile.write(wav_path, FS, (signal * 32767).astype(np.int16))

step = int(FS * 0.01)                           # 每 0.01 秒取一个点
truth = np.column_stack([t[::step], rpm[::step]])
np.savetxt(out_dir / "truth_rpm.csv", truth, delimiter=",",
           header="time_s,rpm", comments="", fmt=["%.2f", "%.0f"])

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t, rpm)
ax.set_xlabel("Time (s)")
ax.set_ylabel("RPM")
ax.set_title("True RPM profile (ground truth)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(out_dir / "rpm_truth.png", dpi=150)

print(f"已生成: {wav_path.name}, truth_rpm.csv, rpm_truth.png")
print(f"时长 {DURATION} 秒，采样率 {FS} Hz，基频范围 {f0.min():.0f}–{f0.max():.0f} Hz")
plt.show()
