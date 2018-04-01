"""
Provides the non-linear functions for processing ECGs. Signals should be input
as an np.array of R-R intervals. Arrays can be multi-dimensional as long as
axis 1 moves through time (i.e. inputs should have shapes (n,) or (m, n)).

Includes function: csi (cardiac sympathatic index) apen (approximate entropy),
spen (spectral entropy), lle (largest Lypunov exponent).
"""

import numpy as np
from scipy.fftpack import fft, ifft


def csi(intervals, num_points=10):
    """ From Geometry of the Poincare plot of RR intervals and its asymmetry in
    healthy adults, J. Piskorski and P. Guzik; and A new method of assessing
    cardiac autonomic function and its comparison with spectral analysis and
    coefficient of variation of R--R interval, Motomi Toichi, Takeshi Sugiura
    Toshiya Murai, and Akira Sengoku.

    Cardiac Sympathetic Index (CSI). The poincare plot is method for visualizing
    chaotic signals by plotting the peaks of a signal against the same peaks
    delayed by one, for use with heart rate the peaks used are the R-R
    intervals. This produces a ellipse aligned along the line x=y with major
    and minor axes 4*SD2 and 4*SD1 respectively. The minor axis represents
    variation between consecutive beats while the major axis represents
    total beat difference. The CSI is given by SD2/SD1. Large CSI values
    indicate relatively large inter-beat variation.

    Parameters:
        num_points (positive integer): The number of datapoints used to
        calculate the CSI. Used as a window so the output will be of length
        len(intervals) - num_points + 1.
    """

    sd1, sd2 = _sd(intervals, num_points)

    sd2[sd1 == 0] = 1
    sd1[sd1 == 0] = 1

    return (sd2 / sd1).T.mean(axis=1)


def _sd(intervals, num_points):
    axis = _function_dimension(intervals)

    signal_length = intervals.shape[axis]
    n = signal_length - num_points + 1

    indices = np.sum(np.mgrid[0:n, 0:num_points], axis=0)

    if axis == 0:
        windowed_intervals = intervals[indices]
    else:
        windowed_intervals = intervals.swapaxes(0, axis)[indices]

    x = windowed_intervals[:-1]
    y = windowed_intervals[1:]

    def _means(x, y):
        mean_x = x.mean(axis=1)
        mean_y = y.mean(axis=1)
        return(mean_x, mean_y)

    def _transpose(vals):
        if axis == 0:
            return vals.T
        else:
            return vals.swapaxes(0, axis)

    mean_x, mean_y = _means(x, y)

    def _sd1():
        mean = _transpose(np.array([mean_y - mean_x]))

        sd1 = np.std((x - y) + mean, axis=1) / (2.0 ** 0.5)
        return sd1

    def _sd2():
        mean = _transpose(np.array([mean_x + mean_y]))

        sd2 = np.std((x + y) - mean, axis=1) / (2.0 ** 0.5)
        return sd2

    return(_sd1(), _sd2())


def apen(intervals, m=2, r=0.6):
    """ Approximate Entropy (ApEn) as described in "Physiological time-series
    analysis what does regularity quantify?" by Steven M. Pingus And Ary L.
    Goldberger.

    Vector x_i contains the ith heart rate to the (i + m - 1)th heart rate.
    The distance between two vectors, x_i and x_j, is greater than r if
    abs(x_i[k] - x_j[k]) > r for any k = 0 ... (m - 1). A pair of vectors
    (or groups), x_i and x_j, are said to be close if the distance between them
    is less than r. C_i is the number of close groups of length m + 1 divided
    by the number of close groups of length m.

    Based on the definition of distance if the ith and jth group are close
    when using length m + 1 then they must also be close when using a length of
    only m. Therefore C is the probability heart rate i + m is close to heart
    rate j + m given all m heart rates in groups i and j are also close.

    ApEn = phi^(m+1)(r) - phi^m(r) where phi^m(r) is the average of natural
    log C_i, for all i groups, calculated using a group size of m.

    Parameters:
        m (positive int): group lengths.
        r (float): max distance between close groups.
    """

    heart_rates = 1 / intervals.astype(np.float32)

    num_close_groups_m = _find_num_close_groups(heart_rates, m, r)
    num_close_groups_m_plus_1 = _find_num_close_groups(heart_rates, m+1, r)
    num_close_groups_m_plus_2 = _find_num_close_groups(heart_rates, m+2, r)

    C_m = num_close_groups_m_plus_1 / num_close_groups_m[:-1]
    C_m_plus_1 = num_close_groups_m_plus_2 / num_close_groups_m_plus_1[:-1]

    phi = lambda C: np.nanmean(np.log(C), axis=0)

    return phi(C_m_plus_1) - phi(C_m)


def _find_num_close_groups(heart_rates, m, r):
    dim = _function_dimension(heart_rates)

    err_msg = 'Group lengths must be smaller than the signal length'
    assert m < heart_rates.shape[dim], err_msg

    if dim == 0:
        dist_mat = _one_dim_distance_matrix(heart_rates)
    else:
        dist_mat = _multi_dim_distance_matrix(heart_rates)

    far_vals = np.logical_or(np.greater(dist_mat, r), np.less(dist_mat, -r))

    return _sum_num_close_groups(far_vals, m).astype(np.float32)


def _function_dimension(x):

    if len(x.shape) > 1:
        return 1
    return 0


def _one_dim_distance_matrix(vals):
    repeats = np.tile(vals, (vals.shape[0], 1))
    return repeats - repeats.T


def _multi_dim_distance_matrix(vals):
    vals = _rotate_and_repeat(vals)
    return vals - np.swapaxes(vals, 0, 1)


def _rotate_and_repeat(vals):
    vals = np.swapaxes(np.array([vals]), 1, 2)
    size = vals.shape
    new_size = (size[1],) + size[1:]
    return np.broadcast_to(vals, new_size)


def _sum_num_close_groups(group_dist_mat, m):
    return np.sum(_is_group_close(group_dist_mat, m), axis=0)


def _is_group_close(far_vals, m):
    close_groups = 0

    for str_idx in xrange(m):
        end_idx = m - str_idx
        close_groups += far_vals[str_idx:-end_idx, str_idx:-end_idx]

    return close_groups == 0


def spen(intervals):
    """ Spectral Entropy (SpEn) is a measure of entropy based on the
    probability mass distribution of the discreate Fourier transformation.
    If a few frequencies dominate a signal the signal is predictable and
    thus has a low entropy. SpEn uses log based 2 and can therefore be
    interpreted as the min number of bits needed to encode the signals
    power spectrum. Because of this length of the signal can affect the
    outcome. """

    axis = _function_dimension(intervals)

    spectrum = np.abs(fft(intervals)) ** 2
    probs = spectrum / np.array([spectrum.sum(axis=axis)]).T

    return - np.sum(probs * np.log2(probs), axis=axis)


def lle(intervals):
    """ Largest Lypunov exponent (LLE) is a measure chaos within a signal.
    If the LLE of a signal is positive the signal is determined to be chaotic.
    The Lypunov exponent of each dimension represents how quickly two initially
    close points move apart from one another.

    This method of calculating the LLE is based on M. Rosenstein, J. Collins,
    and C. De Luca's method from "A practical method for calculating largest
    Lypunov exponents from small data sets".
    """

    dim = _function_dimension(intervals)
    j = _calc_j_from_autocorr(intervals, dim)


def _calc_j_from_autocorr(intervals, axis):

    Intervals = fft(intervals, axis=axis)
    Corr = np.abs(Intervals ** 2)
    corr = ifft(Corr, axis=axis).real

    if axis is 0:
        corr = corr[:len(corr) / 2] / corr[0]
    else:
        corr = corr[:, :corr.shape[1] / 2] / np.array([corr[:, 0]]).T

    diminish_factor = 1 - 1/np.exp(1)
    lag_vals = np.abs(corr - diminish_factor)
    min_val = np.argmin(lag_vals, axis=axis)

    return min_val
