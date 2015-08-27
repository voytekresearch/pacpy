# -*- coding: utf-8 -*-
"""
Functions to calculate phase-amplitude coupling.
"""
from __future__ import division
import numpy as np
from scipy.signal import hilbert
from scipy.stats.mstats import zscore
from pacpy.filt import firf, butterf, morletT
import statsmodels.api as sm


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


def plv(lo, hi, f_lo, f_hi, fs=1000, filterfn=None, filter_kwargs=None):
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
    filterfn : function
        The filtering function, `filterfn(x, f_range, filter_kwargs)`
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`

    Returns
    -------
    pac : scalar
        PAC value
    """

    # Arg check
    _x_sanity(lo, hi)
    _range_sanity(f_lo, f_hi)

    # Filter setup
    if filterfn is None:
        filterfn = firf
    
    if filter_kwargs is None:
        filter_kwargs = {}

    # Filter
    xlo = filterfn(lo, f_lo, fs, **filter_kwargs)
    xhi = filterfn(hi, f_hi, fs, **filter_kwargs)
    amp = np.abs(hilbert(xhi))
    xhiamplo = filterfn(amp, f_lo, fs, **filter_kwargs)

    # Calculate PLV
    pha1 = np.angle(hilbert(xlo))
    pha2 = np.angle(hilbert(xhiamplo))
    pac = np.abs(np.sum(np.exp(1j * (pha1 - pha2)))) / len(xlo)

    return pac


def mi_tort(lo, hi, f_lo, f_hi, fs=1000, Nbins=20, filterfn=None, filter_kwargs=None):
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
        The low frequency filtering range
    fs : float
        The sampling rate (default = 1000Hz)
    filterfn : functional
        The filtering function, `filterfn(x, f_range, filter_kwargs)`
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`
    Nbins : int
        Number of bins to split up the low frequency oscillation cycle

    Returns
    -------
    pac : scalar
        PAC value
    """

    # Arg check
    _x_sanity(lo, hi)
    _range_sanity(f_lo, f_hi)
    if np.logical_or(Nbins < 2, Nbins != int(Nbins)):
        raise ValueError('Number of bins in the low frequency oscillation cycle must be an integer >1.')

    # Filter setup
    if filterfn is None:
        filterfn = firf
    
    if filter_kwargs is None:
        filter_kwargs = {}

    # Filter
    lo = filterfn(lo, f_lo, fs, **filter_kwargs)
    hi = filterfn(hi, f_hi, fs, **filter_kwargs)

    # Calculate phase and amplitude time series
    amp = np.abs(hilbert(hi))
    pha = np.angle(hilbert(lo))
    phadeg = np.degrees(pha)
    
    # Calculate PAC    
    binsize = 360 / Nbins
    phase_lo = np.arange(-180, 180, binsize)
    mean_amp = np.zeros(len(phase_lo))
    for b in xrange(len(phase_lo)):
        phaserange = np.logical_and(phadeg >= phase_lo[b],
                                    phadeg < (phase_lo[b] + binsize))
        mean_amp[b] = np.mean(amp[phaserange])

    p_j = np.zeros(len(phase_lo))
    for b in xrange(len(phase_lo)):
        p_j[b] = mean_amp[b] / sum(mean_amp)
        
    h = -np.sum(p_j * np.log10(p_j))
    h_max = np.log10(Nbins)
    pac = (h_max - h) / h_max
    
    return pac
    

def glm(lo, hi, f_lo, f_hi, fs=1000, filterfn=None, filter_kwargs=None):
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
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`

    Returns
    -------
    pac : scalar
        PAC value
    """

    # Arg check
    _x_sanity(lo, hi)
    _range_sanity(f_lo, f_hi)

    # Filter series
    if filterfn is None:
        filterfn = firf
    
    if filter_kwargs is None:
        filter_kwargs = {}

    # Filter
    lo = filterfn(lo, f_lo, fs, **filter_kwargs)
    hi = filterfn(hi, f_hi, fs, **filter_kwargs)

    # Phase and amplitude time series
    amp = np.abs(hilbert(hi))
    pha = np.angle(hilbert(lo))
    
    # First prepare GLM
    y = amp
    X_pre = np.vstack((np.cos(pha), np.sin(pha)))
    X = X_pre.T
    X = sm.add_constant(X, prepend=False)

    # Run GLM
    glm = sm.GLM(y, X)
    res = glm.fit()

    # Calculate PAC from GLM residuals
    pac = 1 - np.sum(res.resid_deviance ** 2) / np.sum(
        (amp - np.mean(amp)) ** 2)
        
    return pac


def mi_canolty(lo, hi, f_lo, f_hi, fs=1000, filterfn=None, filter_kwargs=None):
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
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`
    Returns
    -------
    pac : scalar
      PAC value
    """

    # Arg check
    _x_sanity(lo, hi)
    _range_sanity(f_lo, f_hi)

    # Filter series
    if filterfn is None:
        filterfn = firf
    
    if filter_kwargs is None:
        filter_kwargs = {}

    # Filter
    lo = filterfn(lo, f_lo, fs, **filter_kwargs)
    hi = filterfn(hi, f_hi, fs, **filter_kwargs)

    # PAC
    amp = np.abs(hilbert(hi))
    pha = np.angle(hilbert(lo))
    pac = np.abs(np.mean(amp * np.exp(1j * pha)))
    return pac
    

def ozkurt(lo, hi, f_lo, f_hi, fs=1000, filterfn=None, filter_kwargs=None):
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
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`

    Returns
    -------
    pac : scalar
      PAC value
    """

    # Arg check
    _x_sanity(lo, hi)
    _range_sanity(f_lo, f_hi)

    # Filter series
    if filterfn is None:
        filterfn = firf
    
    if filter_kwargs is None:
        filter_kwargs = {}

    # Filter
    lo = filterfn(lo, f_lo, fs, **filter_kwargs)
    hi = filterfn(hi, f_hi, fs, **filter_kwargs)

    # PAC
    amp = np.abs(hilbert(hi))
    pha = np.angle(hilbert(lo))
    pac = np.abs(np.sum(amp * np.exp(1j * pha))) / (np.sqrt(len(pha)) * np.sqrt(np.sum(amp**2)))
    return pac


def otc(x, f_hi, f_step, fs=1000,
        w=7, event_prc=95, t_modsig=None, t_buffer=.01):
    """
    Calculate the oscillation-triggered coupling measure of phase-amplitude
    coupling from Dvorak, 2014.

    Parameters
    ----------
    x : array-like, 1d
        The time series
    f_hi : (low, high), Hz
        The low frequency filtering range
    f_step : float, Hz
        The width of each frequency bin in the time-frequency representation
    fs : float
        Sampling rate
    w : float
        Length of the filter in terms of the number of cycles of the
        oscillation whose frequency is the center of the bandpass filter
    event_prc : float (in range 0-100)
        The percentile threshold of the power signal of an oscillation
        for an event to be declared
    t_modsig : (min, max)
        Time (seconds) around an event to extract to define the modulation
        signal
    t_buffer : float
        Minimum time (seconds) in between high frequency events

    Returns
    -------
    pac : float
        phase-amplitude coupling value
    tf : 2-dimensional array
        time-frequency representation of input signal
    a_events : array
        samples at which a high frequency event occurs
    mod_sig : array
        modulation signal (see Dvorak, 2014)

    Algorithm (may be changed in the future)
    ---------
    * Calculate time-frequency representation
    * Define time locking events
    * Calculate modulatory signal for each center frequency
    * Calculate modulation strength for each frequency
    * Identify frequency with the max modulation strength, and set PAC
      equal to that maximal modulation strength

    """
    
    # Arg check
    _x_sanity(x, None)
    _range_sanity(None, f_hi)
    # Set default time range for modulatory signal
    if t_modsig == None:
        t_modsig = (-1,1)
    if f_step <= 0:
        raise ValueError('Frequency band width must be a positive number.')
    if t_modsig[0] > t_modsig[1]:
        raise ValueError('Invalid time range for modulation signal.')
    

    # Calculate the time-frequency representation
    f0s = np.arange(f_hi[0], f_hi[1], f_step)
    tf = morletT(x, f0s, w=w, fs=fs)

    # Find the high frequency activity event times
    F = len(f0s)
    a_events = np.zeros(F, dtype=object)
    for f in xrange(F):
        a_events[f] = _peaktimes(zscore(np.abs(tf[f])), prc=event_prc, t_buffer=t_buffer)
    
    # Calculate the modulation signal
    samp_modsig = np.arange(t_modsig[0] * fs, t_modsig[1] * fs)
    samp_modsig = samp_modsig.astype(int)
    S = len(samp_modsig)
    mod_sig = np.zeros([F, S])

    # For each frequency in the time-frequency representation, calculate a modulation signal
    for f in xrange(F):
        # Exclude high frequency events that are too close to the signal
        # boundaries to extract an entire modulation signal
        mask = np.ones(len(a_events[f]), dtype=bool)
        mask[a_events[f] <= samp_modsig[-1]] = False
        mask[a_events[f] >= (len(x) - samp_modsig[-1])] = False
        a_events[f] = a_events[f][mask]
        
        # Calculate the average LFP around each high frequency event
        E = len(a_events[f])
        for e in xrange(E):
            cur_ecog = x[a_events[f][e] + samp_modsig]
            mod_sig[f] = mod_sig[f] + cur_ecog / E

    # Calculate modulation strength, the range of the modulation signal
    mod_strength = np.zeros(F)
    for f in xrange(F):
        mod_strength = np.max(mod_sig[f]) - np.min(mod_sig[f])
    
    # Calculate PAC
    pac = np.max(mod_strength)
    
    return pac, tf, a_events, mod_sig


def _peaktimes(x, prc=95, t_buffer=.01, fs=1000):
    """
    Calculate event times for which the power signal x peaks

    Parameters
    ----------
    x : array
        Time series of power
    prc : float (in range 0-100)
        The percentile threshold of x for an event to be declares
    t_buffer : float
        Minimum time (seconds) in between events
    fs : float
        Sampling rate
    """
    if np.logical_or(prc < 0, prc >= 100):
        raise ValueError('Percentile threshold must be between 0 and 100.')
        
    samp_buffer = np.int(np.round(t_buffer * fs))
    hi = x > np.percentile(x, prc)
    event_intervals = _chunk_time(hi, samp_buffer=samp_buffer)
    E = np.int(np.size(event_intervals) / 2)
    events = np.zeros(E, dtype=object)
    
    for e in xrange(E):
        temp = x[np.arange(event_intervals[e][0], event_intervals[e][1] + 1)]
        events[e] = event_intervals[e][0] + np.argmax(temp)

    return events


def _chunk_time(x, samp_buffer=0):
    """
    Define continuous chunks of integers

    Parameters
    ----------
    x : array
        Array of integers
    samp_buffer : int
        Minimum number of samples between chunks

    Returns
    -------
    chunks : array (#chunks x 2)
        List of the sample bounds for each chunk
    """
    if samp_buffer < 0:
        raise ValueError('Buffer between signal peaks must be a positive number')
    if samp_buffer != int(samp_buffer):
        raise ValueError('Number of samples must be an integer')
        
    if type(x[0]) == np.bool_:
        Xs = np.arange(len(x))
        x = Xs[x]
    X = len(x)

    cur_start = x[0]
    cur_samp = x[0]
    Nchunk = 0
    chunks = []
    for i in xrange(1, X):
        if x[i] > (cur_samp + samp_buffer + 1):
            if Nchunk == 0:
                chunks = [cur_start, cur_samp]
            else:
                chunks = np.vstack([chunks, [cur_start, cur_samp]])

            Nchunk = Nchunk + 1
            cur_start = x[i]

        cur_samp = x[i]

    # Add final row to chunk
    if Nchunk == 0:
        chunks = [[cur_start, cur_samp]]
    else:
        chunks = np.vstack([chunks, [cur_start, cur_samp]])

    return chunks


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
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`
        
    Returns
    -------
    comod : array-like, 2d
        Matrix of phase-amplitude coupling values for each combination of the
        phase frequency bin and the amplitude frequency bin
    """
    
    # Arg check
    _x_sanity(lo, hi)
    _range_sanity(p_range, a_range)
    if dp <= 0:
        raise ValueError('Width of lo frequqnecy range must be positive')
    if da <= 0:
        raise ValueError('Width of hi frequqnecy range must be positive')

    # Calculate palette frequency parameters
    f_phases = np.arange(p_range[0], p_range[1], dp)
    f_amps = np.arange(a_range[0], a_range[1], da)
    P = len(f_phases)
    A = len(f_amps)

    # Calculate PAC for every combination of P and A
    comod = np.zeros((P, A))
    for p in xrange(P):
        f_lo = (f_phases[p], f_phases[p] + dp)
        
        for a in xrange(A):
            f_hi = (f_amps[a], f_amps[a] + da)
            
            if pac_method == 'plv':
                comod[p,a] = plv(lo, hi, f_lo, f_hi, fs=fs,
                                 filterfn=filterfn, filter_kwargs=filter_kwargs)
            elif pac_method == 'mi_tort':
                comod[p,a] = mi_tort(lo, hi, f_lo, f_hi, fs=fs,
                                 filterfn=filterfn, filter_kwargs=filter_kwargs)
            elif pac_method == 'mi_canolty':
                comod[p,a] = mi_canolty(lo, hi, f_lo, f_hi, fs=fs,
                                 filterfn=filterfn, filter_kwargs=filter_kwargs)
            elif pac_method == 'ozkurt':
                comod[p,a] = ozkurt(lo, hi, f_lo, f_hi, fs=fs,
                                 filterfn=filterfn, filter_kwargs=filter_kwargs)
            elif pac_method == 'glm':
                comod[p,a] = glm(lo, hi, f_lo, f_hi, fs=fs,
                                 filterfn=filterfn, filter_kwargs=filter_kwargs)
            else:
                raise ValueError('PAC method given is invalid.')

    return comod


def pa_series(lo, hi, f_lo, f_hi, fs=1000, filterfn=None, filter_kwargs=None):
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

    Returns
    -------
    pha : array-like, 1d
        Time series of phase
    amp : array-like, 1d
        Time series of amplitude
    """

    # Arg check
    _x_sanity(lo, hi)
    _range_sanity(f_lo, f_hi)

    # Filter setup
    if filterfn is None:
        filterfn = firf
        filter_kwargs = {}

    # Filter
    xlo = filterfn(lo, f_lo, fs, **filter_kwargs)
    xhi = filterfn(hi, f_hi, fs, **filter_kwargs)

    # Calculate phase time series and amplitude time series
    pha = np.angle(hilbert(xlo))
    amp = np.abs(hilbert(xhi))

    return pha, amp


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
        Average
    """
    if np.logical_or(Nbins < 2, Nbins != int(Nbins)):
        raise ValueError('Number of bins in the low frequency oscillation cycle must be an integer >1.')
    if len(pha) != len(amp):
        raise ValueError('Phase and amplitude time series must be of same length.')
    
    phase_bins = np.linspace(-np.pi, np.pi, int(Nbins + 1))
    dist = np.zeros(int(Nbins))

    for b in xrange(int(Nbins)):
        t_phase = np.logical_and(pha >= phase_bins[b],
                                 pha < phase_bins[b + 1])
        dist[b] = np.mean(amp[t_phase])

    return dist