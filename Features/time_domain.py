"""
Provides the linear time-domain functions for processing ECGs. Signals
should be input as an np.array of R-R intervals. Arrays can be
multi-dimensional as long as axis 1 moves through time (i.e. inputs should
have shapes (n,) or (m, n)).

Includes functions: mean, rmssd, sdnn, sdsd, and pnn.
"""

import numpy as np

def _function_dimension(x):

    if len(x.shape) > 1:
        return 1
    return 0


def mean(intervals):
    """ Mean of R-R intervals. """

    axis = _function_dimension(intervals)
    return intervals.mean(axis=axis)


def rmssd(intervals):
    """ Root mean square of successive differences of R-R intervals. """

    axis = _function_dimension(intervals)
    differences = np.diff(intervals, axis=axis)
    rms = np.mean(differences ** 2, axis=axis) ** 0.5

    return rms


def sdnn(intervals):
    """ Standard deviation of R-R intervals. """

    axis = _function_dimension(intervals)
    return intervals.std(axis=axis, ddof=1)


def sdsd(intervals):
    """ Standard deviation of the successive differences of R-R intervals. """

    axis = _function_dimension(intervals)
    differences = np.diff(intervals, axis=axis)
    return differences.std(axis=axis, ddof=1)


def pnn(intervals, min_time=10):
    """ Percent of successive R-R interval differences greater than min_time
    (in same units as R-R intervals). Default min_time is intended to be in
    ms. """

    axis = _function_dimension(intervals)

    differences = np.diff(intervals, axis=axis)
    large_values = np.abs(differences) > min_time
    total_values = differences.shape[axis]

    return large_values.sum(axis=axis)/float(total_values)
