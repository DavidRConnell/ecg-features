"""
Provides the linear frequency-domain functions for processing ECGs. Signals
should be input as an np.array of R-R intervals. Arrays can be
multi-dimensional as long as axis 1 moves through time (i.e. inputs should
have shapes (n,) or (m, n)).

Includes function: psa (power spectral analysis).
"""

from scipy.fftpack import fft, fftfreq
import numpy as np

LF_RANGE = (0.04, 0.15)
HF_RANGE = (0.15, 0.4)

def psa(intervals, sampling_freq):
    """ Power spectral analysis. Gives the ratio of low frequency (0.04 -
    0.15 Hz) to high frequency (0.15 - 0.4 Hz) components of R-R intervals. """

    axis = _function_dimension(intervals)
    size = intervals.shape[axis]
    period = 1.0 / sampling_freq

    power_spec = np.abs(fft(intervals, axis=axis)) ** 2
    freqs = fftfreq(size, d=period)
    LF = np.sum(power_spec.T[_between(freqs, *LF_RANGE)].T, axis=axis)
    HF = np.sum(power_spec.T[_between(freqs, *HF_RANGE)].T, axis=axis)

    return LF/HF

def _function_dimension(x):

    if len(x.shape) > 1:
        return 1
    return 0

def _between(array, lower_bound, upper_bound):
    return np.logical_and(array >= lower_bound, array <= upper_bound)
