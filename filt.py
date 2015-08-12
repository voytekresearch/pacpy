from __future__ import division
import numpy as np
import math
import statsmodels.api as sm

from scipy import stats
from scipy.signal import filtfilt
from scipy.signal import firwin2
from scipy.signal import morlet
from scipy.signal import hilbert

# 1. Filter? - good defaults
#  - Clean - good defaults
# 2. PAC - no defaults for freq; make the user say what they mean.
# Make 1 and 2 composable.

# TODO SC The filterfn signature should be
# filterfn(x, f_range, fs, **filter_kawrgs)
def pac_filter(x_pha, x_amp = None, fs = 1000,
               flo = (13, 30), fhi = (80, 200),
               pac_method = 'mi_tort', filt_method = 'fir',
               w_lo = 7, w_hi = 7, w_hi_flo = False, tw = .15):
    '''
    Filter the time series for PAC calculation

    Parameters
    ----------
    x_pha : array-like 1d
        Time series containing the oscillation whose phase is modulating
    x_amp : array-like 1d
        Time series containing the oscillation whose amplitude is modulated
        Must be the same length as x_pha
        If None: set equal to x_pha
    Fs : float
        Sampling rate of the data in x_pha and x_amp
    flo : 2-element list
        Low and High cutoff frequencies for the modulating oscillation (Hz)
    fhi : 2-element list
        Low and High cutoff frequencies for the modulated oscillation (Hz)
    pac_method : string
        Method to calculate PAC.
        'mi_tort' - See Tort, 2008
        'plv' - See Penny, 2010
        'glm' - See Penny, 2008
        'mi_canolty' - See Canolty, 2006
    filt_method : string
        Method to filter the time series ('morlet' or 'fir')
    w_lo : float
        Length of the filter for the 'flo' range in terms of the number of
        cycles of the center frequency of 'flo'
    w_hi : float
        Length of the filter for the 'fhi' range in terms of the number of
        cycles of the center frequency of 'fhi'
    w_hi_flo : boolean
        If False, the length of the filter for 'fhi' in terms of 'w_hi' is
        actually in terms of the center frequency of 'flo'

    Returns
    -------
    xlo : array
        filtered signal for the oscillation whose phase is modulating
    xhi : array
        filtered signal for the oscillation whose amplitude is modulated
    xhiamplo (optional) : array
        amplitude of xhi filtered in the range of the modulating oscillation
        Used to calculate PAC using the PLV method
    '''
    # Define the time series used for the modulated oscillation
    if x_amp == None:
        x_amp = x_pha
    if len(x_amp) != len(x_pha):
        ValueError('Length of the two temporal signals must be the same')

    # Apply the filters
    if filt_method == 'morlet':
        xlo, M_lo = morlet(x_pha, np.mean(flo), fs = fs, w = w_lo, outM = True)
        if w_hi_flo:
            ValueError('w_hi_flo option not valid for morlet filtering.')
        else:
            xhi = morlet(x_amp, np.mean(fhi), fs = fs, w = w_hi)

    elif filt_method == 'fir':
        xlo, Ntaps_lo = fir(x_pha, frange = flo, fs = fs, w = w_lo, tw = tw, outNtaps = True)
        if w_hi_flo:
            xhi = fir(x_amp, frange = fhi, fs = fs, tw = tw, Ntaps = Ntaps_lo)
        else:
            xhi = fir(x_amp, frange = fhi, fs = fs, w = w_hi, tw = tw)
    else:
        ValueError('Not a valid filtering method')

    # Remove edge artifacts
    xlo = rmv_edgeart(xlo, w_lo, np.mean(flo), fs)
    xhi = rmv_edgeart(xhi, w_lo, np.mean(flo), fs)

    # Filter again if PLV method
    if pac_method == 'plv':
        amp = np.abs(hilbert(xhi))

        if filt_method == 'morlet':
            xhiamplo = filt_morlet(amp, np.mean(flo), fs = fs, w = w_lo)
        elif filt_method == 'fir':
            xhiamplo = fir(amp, frange = flo, fs = fs, w = w_lo, tw = tw)
        else:
            ValueError('Not a valid filtering method')

        return xlo, xhi, xhiamplo

    else:
        return xlo, xhi


def morlet(x, f0, fs = 1000, w = 7, s = 1,
                outM = False, M = None):
    '''
    Filter using a morlet wavelet

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
    outM : boolean
        If True, a second output is returned which is equal to the length of
        the filter used
    M : integer
        Length of the filter. Overrides the f0 and w inputs

    Returns
    -------
    x_filt : array
        Filtered time series
    M (optional) : integer
        Length of the filter
    '''
    raise NotImplementedError('For some reason this is not working. Use FIR filtering')

    if M == None:
        M = 2 * s * w * fs / f0

    morlet_f = morlet(M, w = w, s = s)
    morlet_f = morlet_f / np.sqrt(np.sum(np.abs(morlet_f)**2))
    x_filt = np.convolve(x, np.real(morlet_f), mode = 'same')

    if outM:
        return x_filt, M
    else:
        return x_filt

def fir(x, frange = (13,30), fs = 1000, w = 7, tw = .15,
             outNtaps = False, Ntaps = None):
    '''
    Filter signal with an FIR filter

    x : array
        Time series to filter
    frange : 2-element list
        Cutoff frequencies of bandpass filter
    Fs : float
        Sampling rate
    w : float
        Length of the filter in terms of the number of cycles of the oscillation
        whose frequency is the center of the bandpass filter
    tw : float
        Transition width of the filter in normalized frequency space
    outNtaps : boolean
        If True, a second output is returned which is equal to the length of
        the filter used
    Ntaps : integer
        Length of the filter. Overrides the frange and w inputs

    Returns
    -------
    x_filt : array
        Filtered time series
    M (optional) : integer
        Length of the filter
    '''

    cf = np.mean(frange)
    nyq = fs/2

    if Ntaps == None:
        Ntaps = np.floor(w*fs/cf)

    f = [0, (1-tw)*frange[0]/nyq, frange[0]/nyq, frange[1]/nyq, (1+tw)*frange[1]/nyq, 1]
    m = [0,0,1,1,0,0]
    taps = firwin2(Ntaps, f, m)

    if outNtaps:
        return filtfilt(taps,[1],x), Ntaps
    else:
        return filtfilt(taps,[1],x)


def rmv_edgeart(x,w,cf,fs):
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

