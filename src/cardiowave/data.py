"""Reproducible synthetic waveform generator. No real patient data."""
import numpy as np

CLASSES = ("regular", "early_beat_pattern", "irregular_timing")

def generate(seed=20261030, subjects=60, windows_per_class=4, sample_rate=100, seconds=8):
    rng = np.random.default_rng(seed)
    t = np.arange(sample_rate * seconds) / sample_rate
    xs, ys, gs = [], [], []
    for subject in range(subjects):
        gain = rng.uniform(.8, 1.2)
        width = rng.uniform(.025, .05)
        phase = rng.uniform(0, 2*np.pi)
        for label in range(3):
            for _ in range(windows_per_class):
                rr = rng.uniform(.78, .95)
                beats = [rng.uniform(.25, .55)]
                while beats[-1] < seconds + .5:
                    k = len(beats)
                    if label == 0:
                        interval = rr + rng.normal(0, .018)
                    elif label == 1:
                        interval = rr * (.55 if k % 5 == 2 else 1.45 if k % 5 == 3 else 1)
                    else:
                        interval = rng.uniform(.42, 1.35)
                    beats.append(beats[-1] + max(.25, interval))
                signal = .02*np.sin(2*np.pi*.3*t + phase)
                for beat in beats:
                    signal += gain*np.exp(-.5*((t-beat)/width)**2)
                    signal -= .16*gain*np.exp(-.5*((t-beat+.055)/(width*.8))**2)
                    signal += .24*gain*np.exp(-.5*((t-beat-.22)/.085)**2)
                signal += rng.normal(0, rng.uniform(.008, .03), len(t))
                xs.append(signal.astype(np.float32))
                ys.append(label)
                gs.append(subject)
    return np.stack(xs), np.array(ys), np.array(gs)

def group_split(groups, seed=42, train_fraction=.67, val_fraction=.16):
    ids = np.random.default_rng(seed).permutation(np.unique(groups))
    n_train = max(1, int(len(ids)*train_fraction))
    n_val = max(1, int(len(ids)*val_fraction))
    if n_train+n_val >= len(ids):
        raise ValueError("At least one test group is required")
    parts = (ids[:n_train], ids[n_train:n_train+n_val], ids[n_train+n_val:])
    return tuple(np.flatnonzero(np.isin(groups, p)) for p in parts)
