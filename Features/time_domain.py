"""
Provides the linear time-domain functions for processing ECGs. Signals
should be input as an np.array of R-R intervals. Arrays can be
multi-dimensional as long as axis 1 moves through time (i.e. inputs should
have shapes (n,) or (m, n)).

Provides functions: mean, RMSSD, SDNN, SDSD, and pNN.
"""

import numpy as np

def _function_dimension(x):

    if len(x.shape) > 1:
        return 1
    return 0


def mean(intervals):
    """ Mean of intervals. """

    axis = _function_dimension(intervals)
    return intervals.mean(axis=axis)


def RMSSD(intervals):
    """ Root mean square of successive differences. """

    axis = _function_dimension(intervals)
    differences = np.diff(intervals, axis=axis)
    rms = np.mean(differences ** 2, axis=axis) ** 0.5

    return rms


def SDNN(intervals):
    """ Standard deviation of intervals. """

    axis = _function_dimension(intervals)
    return intervals.std(axis=axis, ddof=1)


def SDSD(intervals):
    """ Standard deviation of the successive differences. """

    axis = _function_dimension(intervals)
    differences = np.diff(intervals, axis=axis)
    return differences.std(axis=axis, ddof=1)


def pNN(intervals, min_time=10):
    """ Percent of successive differences greater than min_time (in same units
    as R-R intervals). """

    axis = _function_dimension(intervals)

    differences = np.diff(intervals, axis=axis)
    large_values = np.abs(differences) > min_time
    total_values = differences.shape[axis]

    return large_values.sum(axis=axis)/float(total_values)
