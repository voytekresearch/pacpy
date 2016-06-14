# -*- coding: utf-8 -*-
"""
Functions to calculate phase-amplitude coupling.
"""
from __future__ import division
import numpy as np
import scipy as sp
from pacpy.filt import firmorlet, firf


def _x_sanity(lo=None, hi=None):
    if lo is not None:
        if np.any(np.isnan(lo)):
            raise ValueError("lo contains NaNs")

    if hi is not None:
        if np.any(np.isnan(hi)):
            raise ValueError("hi contains NaNs")

    if (hi is not None) and (lo is not None):
        if lo.size != hi.size:
            raise ValueError("lo and hi must be the same length")


def _range_sanity(f_lo=None, f_hi=None):
    if f_lo is not None:
        if len(f_lo) != 2:
            raise ValueError("f_lo must contain two elements")

        if f_lo[0] < 0:
            raise ValueError("Elements in f_lo must be > 0")

    if f_hi is not None:
        if len(f_hi) != 2:
            raise ValueError("f_hi must contain two elements")
        if f_hi[0] < 0:
            raise ValueError("Elements in f_hi must be > 0")


def plv(lo, hi, f_lo, f_hi, fs=1000,
        filterfn=None, filter_kwargs=None):
    """
    Calculate PAC using the phase-locking value (PLV) method from prefiltered
    signals

    Parameters
    ----------
    lo : array-like, 1d
        The low frequency time-series to use as the phase component
    hi : array-like, 1d
        The high frequency time-series to use as the amplitude component
    f_lo : (low, high), Hz
        The low frequency filtering range
    f_hi : (low, high), Hz
        The low frequency filtering range
    fs : float
        The sampling rate (default = 1000Hz)
    filterfn : function, False
        The filtering function, `filterfn(x, f_range, filter_kwargs)`

        False activates 'EXPERT MODE'.
        - DO NOT USE THIS FLAG UNLESS YOU KNOW WHAT YOU ARE DOING!
        - In expert mode the user needs to filter the data AND apply the
        hilbert transform.
        - This requires that 'lo' be the phase time series of the low-bandpass
        filtered signal, and 'hi' be the phase time series of the low-bandpass
        of the amplitude of the high-bandpass of the original signal.
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`

    Returns
    -------
    pac : scalar
        PAC value

    Usage
    -----
    >>> import numpy as np
    >>> from scipy.signal import hilbert
    >>> from pacpy.pac import plv
    >>> t = np.arange(0, 10, .001) # Define time array
    >>> lo = np.sin(t * 2 * np.pi * 6) # Create low frequency carrier
    >>> hi = np.sin(t * 2 * np.pi * 100) # Create modulated oscillation
    >>> hi[np.angle(hilbert(lo)) > -np.pi*.5] = 0 # Clip to 1/4 of cycle
    >>> plv(lo, hi, (4,8), (80,150)) # Calculate PAC
    0.99863308613553081
    """

    lo, hi = pa_series(lo, hi, f_lo, f_hi, fs=fs,
                       filterfn=filterfn, filter_kwargs=filter_kwargs, hi_phase=True)

    # Calculate PLV
    pac = np.abs(np.mean(np.exp(1j * (lo - hi))))

    return pac


def _trim_edges(lo, hi):
    """
    Remove extra edge artifact from the signal with the shorter filter
    so that its time series is identical to that of the filtered signal
    with a longer filter.
    """
    
    if len(lo) == len(hi):
        return lo, hi  # Die early if there's nothing to do.
    elif len(lo) < len(hi):
        Ndiff = len(hi) - len(lo)
        if Ndiff % 2 != 0:
            raise ValueError(
                'Difference in filtered signal lengths should be even')
        hi = hi[np.int(Ndiff / 2):np.int(-Ndiff / 2)]
    else:
        Ndiff = len(lo) - len(hi)
        if Ndiff % 2 != 0:
            raise ValueError(
                'Difference in filtered signal lengths should be even')                
        lo = lo[np.int(Ndiff / 2):np.int(-Ndiff / 2)]

    return lo, hi


def mi_tort(lo, hi, f_lo, f_hi, fs=1000,
            Nbins=20, filterfn=None, filter_kwargs=None):
    """
    Calculate PAC using the modulation index method from prefiltered
    signals

    Parameters
    ----------
    lo : array-like, 1d
        The low frequency time-series to use as the phase component
    hi : array-like, 1d
        The high frequency time-series to ue as the amplitude component
    f_lo : (low, high), Hz
        The low frequency filtering ranges
    f_hi : (low, high), Hz
        The low frequency filtering range=
    fs : float
        The sampling rate (default = 1000Hz)
    filterfn : functional
        The filtering function, `filterfn(x, f_range, filter_kwargs)`

        False activates 'EXPERT MODE'. 
        - DO NOT USE THIS FLAG UNLESS YOU KNOW WHAT YOU ARE DOING! 
        - In expert mode the user needs to filter the data AND apply the 
        hilbert transform. 
        - This requires that 'lo' be the phase time series of the low-bandpass
        filtered signal, and 'hi' be the amplitude time series of the high-
        bandpass of the original signal.
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`
    Nbins : int
        Number of bins to split up the low frequency oscillation cycle

    Returns
    -------
    pac : scalar
        PAC value

    Usage
    -----
    >>> import numpy as np
    >>> from scipy.signal import hilbert
    >>> from pacpy.pac import mi_tort
    >>> t = np.arange(0, 10, .001) # Define time array
    >>> lo = np.sin(t * 2 * np.pi * 6) # Create low frequency carrier
    >>> hi = np.sin(t * 2 * np.pi * 100) # Create modulated oscillation
    >>> hi[np.angle(hilbert(lo)) > -np.pi*.5] = 0 # Clip to 1/4 of cycle
    >>> mi_tort(lo, hi, (4,8), (80,150)) # Calculate PAC
    0.34898478944110811
    """

    # Arg check
    if np.logical_or(Nbins < 2, Nbins != int(Nbins)):
        raise ValueError(
            'Number of bins in the low frequency oscillation cycle'
            'must be an integer >1.')

    lo, hi = pa_series(lo, hi, f_lo, f_hi, fs=fs,
                       filterfn=filterfn, filter_kwargs=filter_kwargs)

    # Convert the phase time series from radians to degrees
    phadeg = np.degrees(lo)

    # Calculate PAC
    binsize = 360 / Nbins
    phase_lo = np.arange(-180, 180, binsize)
    mean_amp = np.zeros(len(phase_lo))
    for b in range(len(phase_lo)):
        phaserange = np.logical_and(phadeg >= phase_lo[b],
                                    phadeg < (phase_lo[b] + binsize))
        mean_amp[b] = np.mean(hi[phaserange])

    p_j = np.zeros(len(phase_lo))
    for b in range(len(phase_lo)):
        p_j[b] = mean_amp[b] / sum(mean_amp)

    h = -np.sum(p_j * np.log10(p_j))
    h_max = np.log10(Nbins)
    pac = (h_max - h) / h_max

    return pac


def _ols(y, X):
    """Custom OLS (to minimize outside dependecies)"""

    dummy = np.repeat(1.0, X.shape[0])
    X = np.hstack([X, dummy[:, np.newaxis]])

    beta_hat, resid, _, _ = np.linalg.lstsq(X, y)
    y_hat = np.dot(X, beta_hat)

    return y_hat, beta_hat


def glm(lo, hi, f_lo, f_hi, fs=1000,
        filterfn=None, filter_kwargs=None):
    """
    Calculate PAC using the generalized linear model (GLM) method

    Parameters
    ----------
    lo : array-like, 1d
        The low frequency time-series to use as the phase component
    hi : array-like, 1d
        The high frequency time-series to use as the amplitude component
    f_lo : (low, high), Hz
        The low frequency filtering range
    f_high : (low, high), Hz
        The low frequency filtering range
    fs : float
        The sampling rate (default = 1000Hz)
    filterfn : functional
        The filtering function, `filterfn(x, f_range, filter_kwargs)`

        False activates 'EXPERT MODE'. 
        - DO NOT USE THIS FLAG UNLESS YOU KNOW WHAT YOU ARE DOING! 
        - In expert mode the user needs to filter the data AND apply the 
        hilbert transform. 
        - This requires that 'lo' be the phase time series of the low-bandpass
        filtered signal, and 'hi' be the amplitude time series of the high-
        bandpass of the original signal.
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`

    Returns
    -------
    pac : scalar
        PAC value

    Usage
    -----
    >>> import numpy as np
    >>> from scipy.signal import hilbert
    >>> from pacpy.pac import glm
    >>> t = np.arange(0, 10, .001) # Define time array
    >>> lo = np.sin(t * 2 * np.pi * 6) # Create low frequency carrier
    >>> hi = np.sin(t * 2 * np.pi * 100) # Create modulated oscillation
    >>> hi[np.angle(hilbert(lo)) > -np.pi*.5] = 0 # Clip to 1/4 of cycle
    >>> glm(lo, hi, (4,8), (80,150)) # Calculate PAC
    0.69090396896138917
    """

    lo, hi = pa_series(lo, hi, f_lo, f_hi, fs=fs,
                       filterfn=filterfn, filter_kwargs=filter_kwargs)

    # First prepare GLM
    y = hi
    X_pre = np.vstack((np.cos(lo), np.sin(lo)))
    X = X_pre.T
    y_hat, beta_hat = _ols(y, X)
    resid = y - y_hat

    # Calculate PAC from GLM residuals
    pac = 1 - np.sum(resid ** 2) / np.sum(
        (hi - np.mean(hi)) ** 2)

    return pac


def mi_canolty(lo, hi, f_lo, f_hi, fs=1000,
               filterfn=None, filter_kwargs=None, n_surr=100):
    """
    Calculate PAC using the modulation index (MI) method defined in Canolty,
    2006

    Parameters
    ----------
    lo : array-like, 1d
        The low frequency time-series to use as the phase component
    hi : array-like, 1d
        The high frequency time-series to use as the amplitude component
    f_lo : (low, high), Hz
        The low frequency filtering range
    f_hi : (low, high), Hz
        The low frequency filtering range
    fs : float
        The sampling rate (default = 1000Hz)
    filterfn : functional
        The filtering function, `filterfn(x, f_range, filter_kwargs)`

        False activates 'EXPERT MODE'. 
        - DO NOT USE THIS FLAG UNLESS YOU KNOW WHAT YOU ARE DOING! 
        - In expert mode the user needs to filter the data AND apply the 
        hilbert transform. 
        - This requires that 'lo' be the phase time series of the low-bandpass
        filtered signal, and 'hi' be the amplitude time series of the high-
        bandpass of the original signal.
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`
    n_surr : int
        Number of surrogate tests to run to calculate normalized MI

    Returns
    -------
    pac : scalar
      PAC value

    Usage
    -----
    >>> import numpy as np
    >>> from scipy.signal import hilbert
    >>> from pacpy.pac import mi_canolty
    >>> t = np.arange(0, 10, .001) # Define time array
    >>> lo = np.sin(t * 2 * np.pi * 6) # Create low frequency carrier
    >>> hi = np.sin(t * 2 * np.pi * 100) # Create modulated oscillation
    >>> hi[np.angle(hilbert(lo)) > -np.pi*.5] = 0 # Clip to 1/4 of cycle
    >>> mi_canolty(lo, hi, (4,8), (80,150)) # Calculate PAC
    1.1605177063713188
    """

    lo, hi = pa_series(lo, hi, f_lo, f_hi, fs=fs,
                       filterfn=filterfn, filter_kwargs=filter_kwargs)

    # Calculate modulation index
    pac = np.abs(np.mean(hi * np.exp(1j * lo)))

    # Calculate surrogate MIs
    pacS = np.zeros(n_surr)
    
    loj = np.exp(1j * lo)
    for s in range(n_surr):
        loS = np.roll(loj, np.random.randint(len(lo)))
        pacS[s] = np.abs(np.mean(hi * loS))

    # Return z-score of observed PAC compared to null distribution
    return (pac - np.mean(pacS)) / np.std(pacS)


def ozkurt(lo, hi, f_lo, f_hi, fs=1000,
           filterfn=None, filter_kwargs=None):
    """
    Calculate PAC using the method defined in Ozkurt & Schnitzler, 2011

    Parameters
    ----------
    lo : array-like, 1d
        The low frequency time-series to use as the phase component
    hi : array-like, 1d
        The high frequency time-series to use as the amplitude component
    f_lo : (low, high), Hz
        The low frequency filtering range
    f_hi : (low, high), Hz
        The low frequency filtering range
    fs : float
        The sampling rate (default = 1000Hz)
    filterfn : functional
        The filtering function, `filterfn(x, f_range, filter_kwargs)`

        False activates 'EXPERT MODE'. 
        - DO NOT USE THIS FLAG UNLESS YOU KNOW WHAT YOU ARE DOING! 
        - In expert mode the user needs to filter the data AND apply the 
        hilbert transform. 
        - This requires that 'lo' be the phase time series of the low-bandpass
        filtered signal, and 'hi' be the amplitude time series of the high-
        bandpass of the original signal.
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`

    Returns
    -------
    pac : scalar
      PAC value

    Usage
    -----
    >>> import numpy as np
    >>> from scipy.signal import hilbert
    >>> from pacpy.pac import ozkurt
    >>> t = np.arange(0, 10, .001) # Define time array
    >>> lo = np.sin(t * 2 * np.pi * 6) # Create low frequency carrier
    >>> hi = np.sin(t * 2 * np.pi * 100) # Create modulated oscillation
    >>> hi[np.angle(hilbert(lo)) > -np.pi*.5] = 0 # Clip to 1/4 of cycle
    >>> ozkurt(lo, hi, (4,8), (80,150)) # Calculate PAC
    0.48564417921240238
    """

    lo, hi = pa_series(lo, hi, f_lo, f_hi, fs=fs,
                       filterfn=filterfn, filter_kwargs=filter_kwargs)

    # Calculate PAC
    pac = np.abs(np.sum(hi * np.exp(1j * lo))) / \
        (np.sqrt(len(lo)) * np.sqrt(np.sum(hi**2)))
    return pac
    
    
def comodulogram(lo, hi, p_range, a_range, dp, da, fs=1000,
                 pac_method='mi_tort',
                 filterfn=None, filter_kwargs=None):
    """
    Calculate PAC for many small frequency bands

    Parameters
    ----------
    lo : array-like, 1d
        The low frequency time-series to use as the phase component
    hi : array-like, 1d
        The high frequency time-series to use as the amplitude component
    p_range : (low, high), Hz
        The low frequency filtering range
    a_range : (low, high), Hz
        The high frequency filtering range
    dp : float, Hz
        Width of the low frequency filtering range for each PAC calculation
    da : float, Hz
        Width of the high frequency filtering range for each PAC calculation
    fs : float
        The sampling rate (default = 1000Hz)
    pac_method : string
        Method to calculate PAC.
        'mi_tort' - See Tort, 2008
        'plv' - See Penny, 2008
        'glm' - See Penny, 2008
        'mi_canolty' - See Canolty, 2006
        'ozkurt' - See Ozkurt & Schnitzler, 2011
    filterfn : function
        The filtering function, `filterfn(x, f_range, filter_kwargs)`
        In this case, filter functions should NOT remove edge artifact. Edge
        artifacts will be removed based on the frequency and 'w_lo' parameter
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`

    Returns
    -------
    comod : array-like, 2d
        Matrix of phase-amplitude coupling values for each combination of the
        phase frequency bin and the amplitude frequency bin

    Usage
    -----
    >>> import numpy as np
    >>> from scipy.signal import hilbert
    >>> from pacpy.pac import comodulogram
    >>> t = np.arange(0, 10, .001) # Define time array
    >>> lo = np.sin(t * 2 * np.pi * 6) # Create low frequency carrier
    >>> hi = np.sin(t * 2 * np.pi * 100) # Create modulated oscillation
    >>> hi[np.angle(hilbert(lo)) > -np.pi*.5] = 0 # Clip to 1/4 of cycle
    >>> comod = comodulogram(lo, hi, (5,25), (75,175), 10, 50) # Calculate PAC
    >>> print comod
    [[ 0.32708628  0.32188585]
     [ 0.3295994   0.32439953]]
    """

    # Arg check
    _x_sanity(lo, hi)
    _range_sanity(p_range, a_range)
    if dp <= 0:
        raise ValueError('Width of lo frequency range must be positive')
    if da <= 0:
        raise ValueError('Width of hi frequency range must be positive')

    # method check
    method2fun = {'plv': plv, 'mi_tort': mi_tort, 'mi_canolty': mi_canolty,
                  'ozkurt': ozkurt, 'glm': glm}
    pac_fun = method2fun.get(pac_method, None)
    if pac_fun == None:
        raise ValueError('PAC method given is invalid.')
        
    # Filter setup
    if filterfn is None:
        filterfn = firmorlet
    else:
        if filter_kwargs is None:
            filter_kwargs = {}
        

    # Calculate palette frequency parameters
    f_phases = np.arange(p_range[0], p_range[1], dp)
    f_amps = np.arange(a_range[0], a_range[1], da)
    P = len(f_phases)
    A = len(f_amps)
    
    # Calculate all phase time series
    phaseT = np.zeros(P,dtype=object)
    for p in range(P):
        f_lo = (f_phases[p], f_phases[p] + dp)
        loF = filterfn(lo, f_lo, fs, **filter_kwargs)
        phaseT[p] = np.angle(sp.signal.hilbert(loF))
    
    # Calculate all amplitude time series
    ampT = np.zeros(A,dtype=object)
    for a in range(A):
        f_hi = (f_amps[a], f_amps[a] + da)
        hiF = filterfn(hi, f_hi, fs, **filter_kwargs)
        ampT[a] = np.abs(sp.signal.hilbert(hiF))

    # Calculate PAC for every combination of P and A
    comod = np.zeros((P, A))
    for p in range(P):
        for a in range(A):
            pacphase, pacamp = _trim_edges(phaseT[p],ampT[a])
            comod[p, a] = pac_fun(pacphase, pacamp, [], [], fs=fs, filterfn=False)
    return comod


def pa_series(lo, hi, f_lo, f_hi, fs=1000,
              filterfn=None, filter_kwargs=None, hi_phase=False):
    """
    Calculate the phase and amplitude time series

    Parameters
    ----------
    lo : array-like, 1d
        The low frequency time-series to use as the phase component
    hi : array-like, 1d
        The high frequency time-series to use as the amplitude component
    f_lo : (low, high), Hz
        The low frequency filtering range
    f_hi : (low, high), Hz
        The low frequency filtering range
    fs : float
        The sampling rate (default = 1000Hz)
    filterfn : function
        The filtering function, `filterfn(x, f_range, filter_kwargs)`
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`
    hi_phase : boolean
        Whether to calculate phase of low-frequency component of the high frequency
        time-series amplitude instead of amplitude of the high frequency time-series
        (default = False)

    Returns
    -------
    pha : array-like, 1d
        Time series of phase
    amp : array-like, 1d
        Time series of amplitude (or phase of low frequency component of amplitude if hi_phase=True)

    Usage
    -----
    >>> import numpy as np
    >>> from scipy.signal import hilbert
    >>> from pacpy.pac import pa_series
    >>> t = np.arange(0, 10, .001) # Define time array
    >>> lo = np.sin(t * 2 * np.pi * 6) # Create low frequency carrier
    >>> hi = np.sin(t * 2 * np.pi * 100) # Create modulated oscillation
    >>> hi[np.angle(hilbert(lo)) > -np.pi*.5] = 0 # Clip to 1/4 of cycle
    >>> pha, amp = pa_series(lo, hi, (4,8), (80,150))
    >>> print pha
    [ 1.57079633  1.60849544  1.64619455 ...,  1.45769899  1.4953981  1.53309721]
    """

    # Arg check
    _x_sanity(lo, hi)

    # Filter setup
    if filterfn is None:
        filterfn = firf

    if filter_kwargs is None:
        filter_kwargs = {}

    # Filter then hilbert
    if filterfn is not False:
        _range_sanity(f_lo, f_hi)
        lo = filterfn(lo, f_lo, fs, **filter_kwargs)
        hi = filterfn(hi, f_hi, fs, **filter_kwargs)

        lo = np.angle(sp.signal.hilbert(lo))
        hi = np.abs(sp.signal.hilbert(hi))

        # if high frequency should be returned as phase of low-frequency
        # component of the amplitude:
        if hi_phase == True:
            hi = filterfn(hi, f_lo, fs, **filter_kwargs)
            hi = np.angle(sp.signal.hilbert(hi))

        # Make arrays the same size
        lo, hi = _trim_edges(lo, hi)

    return lo, hi


def pa_dist(pha, amp, Nbins=10):
    """
    Calculate distribution of amplitude over a cycle of phases

    Parameters
    ----------
    pha : array
        Phase time series
    amp : array
        Amplitude time series
    Nbins : int
        Number of phase bins in the distribution,
        uniformly distributed between -pi and pi.

    Returns
    -------
    dist : array
        Average amplitude in each phase bins
    phase_bins : array
        The boundaries to each phase bin. Note the length is 1 + len(dist)

    Usage
    -----
    >>> import numpy as np
    >>> from scipy.signal import hilbert
    >>> from pacpy.pac import pa_series, pa_dist
    >>> t = np.arange(0, 10, .001) # Define time array
    >>> lo = np.sin(t * 2 * np.pi * 6) # Create low frequency carrier
    >>> hi = np.sin(t * 2 * np.pi * 100) # Create modulated oscillation
    >>> hi[np.angle(hilbert(lo)) > -np.pi*.5] = 0 # Clip to 1/4 of cycle
    >>> pha, amp = pa_series(lo, hi, (4,8), (80,150))
    >>> phase_bins, dist = pa_dist(pha, amp)
    >>> print dist
    [  7.21154110e-01   8.04347122e-01   4.49207087e-01   2.08747058e-02
       8.03854240e-05   3.45166617e-05   3.45607343e-05   3.51091029e-05
       7.73644631e-04   1.63514941e-01]
    """
    if np.logical_or(Nbins < 2, Nbins != int(Nbins)):
        raise ValueError(
            'Number of bins in the low frequency oscillation cycle must be an integer >1.')
    if len(pha) != len(amp):
        raise ValueError(
            'Phase and amplitude time series must be of same length.')

    phase_bins = np.linspace(-np.pi, np.pi, int(Nbins + 1))
    dist = np.zeros(int(Nbins))

    for b in range(int(Nbins)):
        t_phase = np.logical_and(pha >= phase_bins[b],
                                 pha < phase_bins[b + 1])
        dist[b] = np.mean(amp[t_phase])

    return phase_bins[:-1], dist
