"""Simple, interpretable signal features."""
import numpy as np
from scipy.signal import find_peaks

def extract(signal, sample_rate=100):
    x = np.asarray(signal)
    peaks, _ = find_peaks(x, height=max(.12, float(np.percentile(x,85))),
                          distance=max(1,int(sample_rate*.3)), prominence=.1)
    rr = np.diff(peaks)/sample_rate
    if not len(rr):
        rr = np.array([0.])
    dr = np.diff(rr)
    power = np.abs(np.fft.rfft(x))**2
    p = power/(power.sum()+1e-12)
    return np.array([x.mean(),x.std(),x.min(),x.max(),np.sqrt(np.mean(x*x)),
       np.mean(np.abs(np.diff(x))),len(peaks),rr.mean(),rr.std(),
       np.median(rr),rr.min(),rr.max(),
       np.sqrt(np.mean(dr**2)) if len(dr) else 0.,
       -np.sum(p*np.log(p+1e-12))],dtype=float)

def matrix(signals, sample_rate=100):
    return np.stack([extract(x,sample_rate) for x in signals])
