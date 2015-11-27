from __future__ import division
import numpy as np

from scipy.signal import filtfilt
from scipy.signal import firwin2, firwin
from scipy.signal import morlet
from scipy.signal import butter


def firfls(x, f_range, fs=1000, w=3, tw=.15):
    """
    Filter signal with an FIR filter
    *Like firls in MATLAB

    x : array-like, 1d
        Time series to filter
    f_range : (low, high), Hz
        Cutoff frequencies of bandpass filter
    fs : float, Hz
        Sampling rate
    w : float
        Length of the filter in terms of the number of cycles 
        of the oscillation whose frequency is the low cutoff of the 
        bandpass filter
    tw : float
        Transition width of the filter in normalized frequency space

    Returns
    -------
    x_filt : array-like, 1d
        Filtered time series
    """

    if w <= 0:
        raise ValueError(
            'Number of cycles in a filter must be a positive number.')

    if np.logical_or(tw < 0, tw > 1):
        raise ValueError('Transition width must be between 0 and 1.')

    nyq = fs / 2
    if np.any(np.array(f_range) > nyq):
        raise ValueError('Filter frequencies must be below nyquist rate.')

    if np.any(np.array(f_range) < 0):
        raise ValueError('Filter frequencies must be positive.')

    Ntaps = np.floor(w * fs / f_range[0])
    if len(x) < Ntaps:
        raise RuntimeError(
            'Length of filter is loger than data. ' 
            'Provide more data or a shorter filter.')

    # Characterize desired filter
    f = [0, (1 - tw) * f_range[0] / nyq, f_range[0] / nyq,
         f_range[1] / nyq, (1 + tw) * f_range[1] / nyq, 1]
    m = [0, 0, 1, 1, 0, 0]
    if any(np.diff(f) < 0):
        raise RuntimeError(
            'Invalid FIR filter parameters.'
            'Please decrease the transition width parameter.')

    # Perform filtering
    taps = firwin2(Ntaps, f, m)
    x_filt = filtfilt(taps, [1], x)

    if any(np.isnan(x_filt)):
        raise RuntimeError(
            'Filtered signal contains nans. Adjust filter parameters.')

    return x_filt
    
    
def firf(x, f_range, fs=1000, w=3):
    """
    Filter signal with an FIR filter
    *Like fir1 in MATLAB

    x : array-like, 1d
        Time series to filter
    f_range : (low, high), Hz
        Cutoff frequencies of bandpass filter
    fs : float, Hz
        Sampling rate
    w : float
        Length of the filter in terms of the number of cycles 
        of the oscillation whose frequency is the low cutoff of the 
        bandpass filter

    Returns
    -------
    x_filt : array-like, 1d
        Filtered time series
    """

    if w <= 0:
        raise ValueError(
            'Number of cycles in a filter must be a positive number.')

    nyq = fs / 2
    if np.any(np.array(f_range) > nyq):
        raise ValueError('Filter frequencies must be below nyquist rate.')

    if np.any(np.array(f_range) < 0):
        raise ValueError('Filter frequencies must be positive.')

    Ntaps = np.floor(w * fs / f_range[0])
    if len(x) < Ntaps:
        raise RuntimeError(
            'Length of filter is loger than data. ' 
            'Provide more data or a shorter filter.')

    # Perform filtering
    taps = firwin(Ntaps, f_range)
    x_filt = filtfilt(taps, [1], x)

    if any(np.isnan(x_filt)):
        raise RuntimeError(
            'Filtered signal contains nans. Adjust filter parameters.')

    return x_filt


def butterf(x, f_range, fs=1000, N=2):
    """
    Filter signal with an FIR filter

    x : array-like, 1d
        Time series to filter
    f_range : (low, high), Hz
        Cutoff frequencies of bandpass filter
    fs : float, Hz
        Sampling rate
    N : int
        Order of butteworth filter

    Returns
    -------
    x_filt : array-like, 1d
        Filtered time series
    """

    nyq = fs / 2
    if np.any(np.array(f_range) > nyq):
        raise ValueError('Filter frequencies must be below nyquist rate.')

    if np.any(np.array(f_range) < 0):
        raise ValueError('Filter frequencies must be positive.')

    if np.logical_or(N != int(N), N <= 0):
        raise ValueError('Order of filter must be a positive integer')

    Wn = (f_range[0] / nyq, f_range[1] / nyq)
    b, a = butter(N, Wn, btype='bandpass')
    x_filt = filtfilt(b, a, x)

    if any(np.isnan(x_filt)):
        raise RuntimeError(
            'Filtered signal contains nans. ' 
            'Adjust filter parameters such as decreasing order.')

    return x_filt


def rmvedgeart(x, w, cf, fs):
    """
    Calculate the number of points to remove for edge artifacts

    x : array
        time series to remove edge artifacts from
    cf : float
        low cutoff frequency of the bandpass filter
    w : float
        number of cycles
    Fs : float
        Sampling rate
    """
    win = np.floor(w * fs / cf)
    return x[np.int(win):-np.int(win)]


def morletT(x, f0s, w=3, fs=1000, s=1):
    """
    Calculate the time-frequency representation of the signal 'x' over the
    frequencies in 'f0s' using morlet wavelets

    Parameters
    ----------
    x : array
        time series
    f0s : array
        frequency axis
    w : float
        Length of the filter in terms of the number of cycles 
        of the oscillation whose frequency is the center of the 
        bandpass filter
    Fs : float
        Sampling rate
    s : float
        Scaling factor

    Returns
    -------
    mwt : 2-D array
        time-frequency representation of signal x
    """
    if w <= 0:
        raise ValueError(
            'Number of cycles in a filter must be a positive number.')

    T = len(x)
    F = len(f0s)
    mwt = np.zeros([F, T], dtype=complex)
    for f in range(F):
        mwt[f] = morletf(x, f0s[f], fs=fs, w=w, s=s)

    return mwt


def morletf(x, f0, fs=1000, w=3, s=1, M=None, norm='sss'):
    """
    Convolve a signal with a complex wavelet
    The real part is the filtered signal
    Taking np.abs() of output gives the analytic amplitude
    Taking np.angle() of output gives the analytic phase

    x : array
        Time series to filter
    f0 : float
        Center frequency of bandpass filter
    Fs : float
        Sampling rate
    w : float
        Length of the filter in terms of the number of 
        cycles of the oscillation with frequency f0
    s : float
        Scaling factor for the morlet wavelet
    M : integer
        Length of the filter. Overrides the f0 and w inputs
    norm : string
        Normalization method
        'sss' - divide by the sqrt of the sum of squares of points
        'amp' - divide by the sum of amplitudes divided by 2

    Returns
    -------
    x_trans : array
        Complex time series
    """
    if w <= 0:
        raise ValueError(
            'Number of cycles in a filter must be a positive number.')

    if M == None:
        M = 2 * s * w * fs / f0

    morlet_f = morlet(M, w=w, s=s)
    morlet_f = morlet_f

    if norm == 'sss':
        morlet_f = morlet_f / np.sqrt(np.sum(np.abs(morlet_f)**2))
    elif norm == 'abs':
        morlet_f = morlet_f / np.sum(np.abs(morlet_f)) * 2
    else:
        raise ValueError('Not a valid wavelet normalization method.')

    mwt_real = np.convolve(x, np.real(morlet_f), mode='same')
    mwt_imag = np.convolve(x, np.imag(morlet_f), mode='same')

    return mwt_real + 1j * mwt_imag
