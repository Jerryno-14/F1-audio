# F1 engine RPM from audio

A Python project that estimates engine speed (RPM) from the sound of an F1 onboard recording, and looks for gear shifts as sudden drops in RPM. Work in progress, built while learning Python (NumPy, SciPy, Matplotlib).

## Idea

For a four-stroke V6, the engine fires three times per revolution, so the main sound frequency is about RPM / 20 (for example 12,000 rpm is about 600 Hz). The spectrogram of the recording shows this as a set of harmonic lines, and the spacing between the lines gives the engine speed.

## What the code does

- `gen_synthetic.py` makes a 24 s synthetic engine sound with a known RPM curve (including shifts and noise), so the method can be checked against a known answer. Outputs: `synthetic_engine.wav`, `truth_rpm.csv`, `rpm_truth.png`.
- `estimate_rpm.py` loads a clip, computes a spectrogram (Hann window, 16384 samples, 75% overlap), and tries three estimators in turn:
  1. the loudest frequency (naive, easily fooled by harmonics),
  2. the loudest frequency inside a 40 to 125 Hz band,
  3. a harmonic-spacing score that compares energy on and between harmonic lines (the one used for the result).
  It then removes outliers with a median filter, flags frames where RPM drops by more than 9% as possible gear shifts, and writes `rpm_estimate.csv` and `rpm_with_shifts.png`.

## Result so far

`rpm_with_shifts.png` and `rpm_estimate.csv` come from a 10 s clip of a real onboard recording (starting at 600 s), with one drop flagged as a possible shift. This has not yet been checked against a known answer on real audio, so treat it as a first result, not a measured accuracy.

## Run it

```
python -m venv .venv && source .venv/bin/activate
python -m pip install numpy scipy matplotlib
python gen_synthetic.py
```

To use real audio, extract it from your own video with ffmpeg and set `WAV_FILE`, `START_S` and `LENGTH_S` at the top of `estimate_rpm.py`:

```
ffmpeg -i onboard.mp4 -vn -ac 1 -ar 44100 engine.wav
```

The source video and extracted audio are not included in this repository.

## Next steps

- Run `estimate_rpm.py` on `synthetic_engine.wav` and compare against `truth_rpm.csv`.
- Check the flagged shifts on real audio by ear in Audacity.
