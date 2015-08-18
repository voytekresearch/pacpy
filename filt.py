from __future__ import division
import numpy as np

from scipy.signal import filtfilt
from scipy.signal import firwin2
from scipy.signal import morlet
from scipy.signal import butter

def firf(x, f_range, fs = 1000, w = 7, tw = .15):
    '''
    Filter signal with an FIR filter

    x : array-like, 1d
        Time series to filter
    f_range : (low, high), Hz
        Cutoff frequencies of bandpass filter
    fs : float, Hz
        Sampling rate
    w : float
        Length of the filter in terms of the number of cycles of the oscillation
        whose frequency is the center of the bandpass filter
    tw : float
        Transition width of the filter in normalized frequency space

    Returns
    -------
    x_filt : array-like, 1d
        Filtered time series
    '''

    cf = np.mean(f_range)
    nyq = fs/2
    Ntaps = np.floor(w*fs/cf)

    f = [0, (1-tw)*f_range[0]/nyq, f_range[0]/nyq, f_range[1]/nyq, (1+tw)*f_range[1]/nyq, 1]
    m = [0,0,1,1,0,0]
    taps = firwin2(Ntaps, f, m)
    x_filt = filtfilt(taps,[1],x)

    return x_filt


def butterf(x, f_range, fs = 1000, N = 2):
    '''
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
    '''

    nyq = fs/2
    Wn = (f_range[0]/nyq, f_range[1]/nyq)
    b, a = butter(N, Wn, btype = 'bandpass')
    x_filt = filtfilt(b,a,x)

    return x_filt


def rmv_edgeart(x, w, cf, fs):
    '''
    Calculate the number of points to remove for edge artifacts

    x : array
        time series to remove edge artifacts from
    cf : float
        center frequency of the bandpass filter
    w : float
        number of cycles
    Fs : float
        Sampling rate
    '''
    win = np.floor((w * fs / cf) / 2.0)
    return x[np.int(win):-np.int(win)]


def morletT(x, f0s, w = 7, fs = 1000, s = 1):
    '''
    Calculate the time-frequency representation of the signal 'x' over the
    frequencies in 'f0s' using morlet wavelets

    Parameters
    ----------
    x : array
        time series
    f0s : array
        frequency axis
    w : float
        Length of the filter in terms of the number of cycles of the oscillation
        whose frequency is the center of the bandpass filter
    Fs : float
        Sampling rate
    s : float
        Scaling factor

    Returns
    -------
    mwt : 2-D array
        time-frequency representation of signal x

    '''
    T = len(x)
    F = len(f0s)
    mwt = np.zeros([F,T],dtype=complex)
    for f in range(F):
        mwt[f] = np.abs(trans_morlet(x, f0s[f], fs = fs, w = w, s = s))

    return mwt


def trans_morlet(x, f0, fs = 1000, w = 7, s = 1, M = None):
    '''
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
        Length of the filter in terms of the number of cycles of the oscillation
        with frequency f0
    s : float
        Scaling factor for the morlet wavelet
    M : integer
        Length of the filter. Overrides the f0 and w inputs

    Returns
    -------
    x_trans : array
        Complex time series
    '''
    if M == None:
        M = 2 * s * w * fs / f0

    morlet_f = morlet(M, w = w, s = s)
    morlet_f = morlet_f / np.sqrt(np.sum(np.abs(morlet_f)**2))

    mwt_real = np.convolve(x, np.real(morlet_f), mode = 'same')
    mwt_imag = np.convolve(x, np.imag(morlet_f), mode = 'same')

    return mwt_real + 1j*mwt_imag
