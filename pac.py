# -*- coding: utf-8 -*-
"""
Functions to calculate phase-amplitude coupling.
"""
from __future__ import division
import numpy as np
from pacpy.filt import *


def _x_sanity(lo=None, hi=None):
    if lo is not None:
        if np.any(np.isnan(hi)):
            raise ValueError("hi contains NaNs")

    if hi is not None:
        if np.any(np.isnan(hi)):
            raise ValueError("hi contains NaNs")

    if (hi is not None) and (lo is not None):
        if lo.size != hi.size:
            raise ValueError("lo and must be the same length")


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


def plv(lo, hi, f_hi, f_lo, fs=1000, filterfn=None, filter_kwargs=None):
    """
    Calculate PAC using the phase-locking value (PLV) method from prefiltered
    signals

    Parameters
    ----------
    lo : array-like, 1d
        The low frequency time-series to use as the phase component
    high : array-like, 1d
        The high frequency time-series to use as the amplitude component
    f_lo : (low, high), Hz
        The low frequency filtering range
    f_high : (low, high), Hz
        The low frequency filtering range
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
        filterfn = fir  # TODO or whatever the default name is for plv
        filter_kwargs = {}  # TODO

    # Filter
    lo = filterfn(lo, f_lo, fs, **filter_kwargs)
    hi = filterfn(hi, f_hi, fs, **filter_kwargs)

    # And PAC
    pha = np.angle(hilbert(lo))
    amp = np.angle(hilbert(hi))
    pac = np.abs(np.sum(np.exp(1j * (pha - amp)))) / len(pha)

    return pac


def mi_tort(lo, hi, f_hi, f_lo, fs=1000, filterfn=None, filter_kwargs=None):
    """
    Calculate PAC using the modulation index method from prefiltered
    signals

    Parameters
    ----------
    lo : array-like, 1d
        The low frequency time-series to use as the phase component
    high : array-like, 1d
        The high frequency time-series to use as the amplitude component
    f_lo : (low, high), Hz
        The low frequency filtering range
    f_high : (low, high), Hz
        The low frequency filtering range
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

    # Filter setup
    if filterfn is None:
        filterfn = fir  # TODO or whatever the default name is for mi_tort
        filter_kwargs = {}  # TODO

    # Filter
    lo = filterfn(lo, f_lo, fs, **filter_kwargs)
    hi = filterfn(hi, f_hi, fs, **filter_kwargs)

    # PAC
    amp = np.abs(hilbert(hi))
    pha = np.angle(hilbert(lo))
    phadeg = np.degrees(pha)

    phase_lo = np.arange(-180, 180, 20)
    mean_amp = np.zeros(len(phase_lo))
    for b in range(len(phase_lo)):
        phaserange = np.logical_and(phadeg >= phase_lo[b],
                                    phadeg < (phase_lo[b] + 20))
        mean_amp[b] = np.mean(amp[phaserange])

    p_j = np.zeros(len(phase_lo))
    for b in range(len(phase_lo)):
        p_j[b] = mean_amp[b] / sum(mean_amp)
        h = -np.sum(p_j * np.log10(p_j))
        h_max = np.log10(18)  # TODO explain magic number
        pac = (h_max - h) / h_max

        return pac


def glm(lo, hi, f_hi, f_lo, fs=1000, filterfn=None, filter_kwargs=None):
    """
    Calculate PAC using the generalized linear model (GLM) method

    Parameters
    ----------
    lo : array-like, 1d
        The low frequency time-series to use as the phase component
    high : array-like, 1d
        The high frequency time-series to use as the amplitude component
    f_lo : (low, high), Hz
        The low frequency filtering range
    f_high : (low, high), Hz
        The low frequency filtering range
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
        filterfn = fir  # TODO or whatever the default name is for glm
        filter_kwargs = {}  # TODO

    # Filter
    lo = filterfn(lo, f_lo, fs, **filter_kwargs)
    hi = filterfn(hi, f_hi, fs, **filter_kwargs)

    # PAC
    amp = np.abs(hilbert(hi))
    pha = np.angle(hilbert(lo))

    # First prepare GLM
    y = amp
    X_pre = np.vstack((np.cos(pha), np.sin(pha)))
    X = X_pre.T
    X = sm.add_constant(X, prepend=False)

    # Run GLM
    my_glm = sm.GLM(y, X)
    res = my_glm.fit()
    # print(res.summary())
    # Calculate R^2 value. Equivalent to mdl.Rsquared.Ordinary in MATLAB

    # and actually calculate PAC
    pac = 1 - np.sum(res.resid_deviance ** 2) / np.sum(
        (amp - np.mean(amp)) ** 2)

    return pac


def mi_canolty(lo, hi, f_hi, f_lo, fs=1000, filterfn=None, filter_kwargs=None):
    """
    Calculate PAC using the modulation index (MI) method defined in Canolty,
    2006

    Parameters
    ----------
    lo : array-like, 1d
        The low frequency time-series to use as the phase component
    high : array-like, 1d
        The high frequency time-series to use as the amplitude component
    f_lo : (low, high), Hz
        The low frequency filtering range
    f_high : (low, high), Hz
        The low frequency filtering range
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
        filterfn = fir  # TODO or whatever the default name is for plv
        filter_kwargs = {}  # TODO

    # Filter
    lo = filterfn(lo, f_lo, fs, **filter_kwargs)
    hi = filterfn(hi, f_hi, fs, **filter_kwargs)

    # PAC
    amp = np.abs(hilbert(hi))
    pha = np.angle(hilbert(lo))
    pac = np.abs(np.mean(amp * np.exp(1j * pha)))

    return pac


# TODO remove list arg
# TODO SC can we adapt this to take a filterfn/kwargs, and if we can do
# we want to?
def otc(lohi, f_hi, f_step, w=7, event_prc=95, fs=1000,
        t_modsig=(-1, 1), t_buffer=.01):
    """
    Calculate the oscillation-triggered coupling measure of phase-amplitude
    coupling from Dvorak, 2014.

    Parameters
    ----------
    lohi : array
        time series
    fs : float
        Sampling rate
    f0s : array
        Frequencies in the time-frequency representation
    w : float
        Length of the filter in terms of the number of cycles of the
        oscillation
        whose frequency is the center of the bandpass filter
    event_prc : float (in range 0-100)
        The percentile threshold of the power signal of an oscillation
        for an event to be declared
    t_modsig : array (2x)
        Time (seconds) around an event to extract to define the modulation
        signal
    t_buffer : float
        Minimum time (seconds) in between events

    Returns
    -------
    x_tf : 2-dimensional array
        time-frequency representation of input signal
    a_events : array

    Algorithm
    ---------
    * Calculate time-frequency representation
    * Define time locking events
    * Calculate modulatory signal for each center frequency
    * Calculate modulation strength for each frequency
    * Identify frequency with the max modulation strength, and set PAC
      equal to that maximal modulation strength

    """
    # Arg check
    _x_sanity(lohi, None)
    _range_sanity(None, f_hi)

    f0s = np.arange(f_hi[0], f_hi[1], f_step)

    # Calculate the time-frequency representation
    tf = morletT(lohi, f0s, w=w, fs=fs)

    # z-score and find the high frequency activity event times
    F = len(f0s)
    a_events = np.zeros(F, dtype=object)
    for f in range(F):
        tf[f] = stats.mstats.zscore(tf[f])
        a_events[f] = _peaktimes(tf[f], prc=event_prc, t_buffer=t_buffer)

    # Calculate the modulation signal, its amplitude, frequency, and its
    # phase
    samp_modsig = np.arange(t_modsig[0] * fs, t_modsig[1] * fs)
    samp_modsig = samp_modsig.astype(int)
    S = len(samp_modsig)
    mod_sig = np.zeros([F, S])

    # TODO SC: could you comment this a bit more. Not sure what the hell
    # is a happening.
    for f in range(F):
        mask = np.ones(len(a_events[f]), dtype=bool)
        mask[a_events[f] <= samp_modsig[-1]] = False
        mask[a_events[f] >= (len(lohi) - samp_modsig[-1])] = False
        a_events[f] = a_events[f][mask]
        E = len(a_events[f])
        for e in range(E):
            cur_ecog = lohi[a_events[f][e] + samp_modsig]
            mod_sig[f] = mod_sig[f] + cur_ecog / E

    mod_strength = np.zeros(F)
    for f in range(F):
        mod_strength = np.max(mod_sig[f]) - np.min(mod_sig[f])
    pac = np.max(mod_strength)

    # TODO SC what should this be returning?
    return pac, tf, a_events, mod_sig


def _peaktimes(x, prc=95, t_buffer=.01, fs=1000):
    """
    Calculate event times for which the power signal x peaks

    Parameters
    ----------
    x : array
        time series of power
    prc : float (in range 0-100)
        The percentile threshold of x for an event to be declares
    t_buffer : float
        Minimum time (seconds) in between events
    fs : float
        Sampling rate

    """
    samp_buffer = np.int(np.round(t_buffer * fs))
    hi = x > np.percentile(x, prc)
    event_intervals = _chunk_time(hi, samp_buffer=samp_buffer)
    E = np.int(np.size(event_intervals) / 2)
    events = np.zeros(E, dtype=object)
    for e in range(E):
        temp = x[
            np.arange(event_intervals[e][0], event_intervals[e][1] + 1)]
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

    doctests
    --------
    _chunk_time([5,6,7,8,10,55,56], samp_buffer = 0)
        array([[ 5,  8],
               [10, 10],
               [55, 56]])

    _chunk_time([5,6,7,8,10,55,56], samp_buffer = 2)
        array([[ 5, 10],
               [55, 56]])

    """
    if type(x[0]) == np.bool_:
        Xs = np.arange(len(x))
        x = Xs[x]
    X = len(x)

    cur_start = x[0]
    cur_samp = x[0]
    Nchunk = 0
    chunks = []
    for i in np.arange(1, X):

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
        chunks = [cur_start, cur_samp]
    else:
        chunks = np.vstack([chunks, [cur_start, cur_samp]])

    return chunks


# TODO make mac_method a functional arg. Add a pac_params dist too.
def comodulogram(pha, amp=None, fs=1000, pac_method='mi_tort',
                 dp=2, da=4, p_range=(4, 50), a_range=(10, 200),
                 **kwargs):

    raise NotImplementedError("TODO Adapt to pacpy and move to own submodule?")

    """
    Calculate PAC for many small frequency bands

    Parameters
    ----------
    x_pha : array-like 1d
        Time series containing the oscillation whose phase is modulating
        (flo)
    x_amp : array-like 1d
        Time series containing the oscillation whose amplitude is
        modulated (fhi)
        Must be the same length as x_pha
        If None: set equal to x_pha
    fs : float
        Sampling rate of x_pha and x_amp
    pac_method : string
        Method to calculate PAC.
        'mi_tort' - See Tort, 2008
        'plv' - See Penny, 2010
        'glm' - See Penny, 2008
        'mi_canolty' - See Canolty, 2006
    dp : float
        step size and bandwidth for the low frequency band
    da : float
        step size and bandwidth for the high frequency band
    p_range : array (2x1)
        frequency range for the modulating oscillation (phase frequency)
    a_range : array (2x1)
        frequency range for the modulated oscillation (amplitude frequency)
    **kwargs : dictionary
        Parameters for PAC calculation or filtering

    Returns
    -------
    pacpal : 2D array
        PAC values for each phase and amplitude frequency band
    """

    # Define the time series used for the modulated oscillation
    if amp == None:
        amp = pha
    if len(amp) != len(pha):
        ValueError('Length of the two temporal signals must be the same')

    # Calculate palette frequency parameters
    f_phases = np.arange(p_range[0], p_range[1], dp)
    f_amps = np.arange(a_range[0], a_range[1], da)
    P = len(f_phases)
    A = len(f_amps)

    # Calculate PAC for every combination of P and A
    pacpal = np.zeros((P, A))
    for p in range(P):
        flo = (f_phases[p], f_phases[p] + dp)
        for a in range(A):
            # print p,a
            fhi = (f_amps[a], f_amps[a] + da)
            _, _, pacpal[p, a], _ = pac(pha, amp=amp, fs=fs,
                                        flo=flo, fhi=fhi,
                                        pac_method=pac_method, **kwargs)

    return pacpal


def phaseamp_series(pha, amp=None, fs=1000,
                    flo=(13, 30), fhi=(80, 200), **kwargs):
    """
    Calculate the time series of the phase and the time series of the
    amplitude

    Parameters
    ----------
    x_pha : array-like 1d
        Time series containing the oscillation whose phase is modulating
        (flo)
    x_amp : array-like 1d
        Time series containing the oscillation whose amplitude is
        modulated (fhi)
        Must be the same length as x_pha
        If None: set equal to x_pha
    fs : float
        Sampling rate of x_pha and x_amp
    flo : 2-element list
        Low and High cutoff frequencies for the modulating oscillation (Hz)
    fhi : 2-element list
        Low and High cutoff frequencies for the modulated oscillation (Hz)
    **kwargs : dictionary
        Parameters for PAC calculation or filtering

    Returns
    -------
    pha : array
        Phase time series
    amp : array
        Amplitude time series

    """

    # Define the time series used for the modulated oscillation
    if amp == None:
        amp = pha
    if len(amp) != len(pha):
        ValueError('Length of the two temporal signals must be the same')

    # Filter the signals
    lo, hi = pac_filter(pha, amp, fs=fs, flo=flo, fhi=fhi, **kwargs)

    # Calculate phase time series and amplitude time series
    amp = np.abs(hilbert(hi))
    pha = np.angle(hilbert(lo))

    return pha, amp


def pa_dist(pha, amp, n_bins=10):
    """
    Calculate distribution of amplitude over a cycle of phases

    Parameters
    ----------
    pha : array
        Phase time series
    amp : array
        Amplitude time series
    n_bins : int
        Number of phase bins in the distribution,
        uniformly distributed between -pi and pi.

    Returns
    -------
    dist : array
        Average

    """
    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    dist = np.zeros(n_bins)

    for b in xrange(n_bins):
        t_phase = np.logical_and(pha >= phase_bins[b],
                                 pha < phase_bins[b + 1])
        dist[b] = np.mean(amp[t_phase])

    return dist


#
#
# def pac(x_pha, x_amp = None,  fs = 1000,
#         flo = (13, 30), fhi = (80, 200),
#         pac_method = 'plv', **kwargs):
#     """
#     Calculate phase-amplitude coupling
#
#     Parameters
#     ----------
#     x_pha : array-like 1d
#         Time series containing the oscillation whose phase is
# modulating (flo)
#     x_amp : array-like 1d
#         Time series containing the oscillation whose amplitude is
# modulated (fhi)
#         Must be the same length as x_pha
#         If None: set equal to x_pha
#     fs : float
#         Sampling rate of x_pha and x_amp
#     flo : 2-element list
#         Low and High cutoff frequencies for the modulating oscillation
# (Hz)
#     fhi : 2-element list
#         Low and High cutoff frequencies for the modulated oscillation (Hz)
#     pac_method : string
#         Method to calculate PAC.
#         'mi_tort' - See Tort, 2008
#         'plv' - See Penny, 2010
#         'glm' - See Penny, 2008
#         'mi_canolty' - See Canolty, 2006
#         'otc' - See Dvorak, 2014. Uses function defaults and only 1 signal.
#     **kwargs : dictionary
#         Parameters for PAC calculation or filtering
#
#     Returns
#     -------
#     pac : float
#         Phase-amplitude coupling value
#     """
#
# Define the time series used for the modulated oscillation
#     if x_amp == None:
#         x_amp = x_pha
#     if len(x_amp) != len(x_pha):
#         ValueError('Length of the two temporal signals must be the same')
#
# Filter the signals
#     if pac_method == 'plv':
#         xlo, xhi, xhiamplo = pac_filter(x_pha, x_amp, fs = fs, flo = flo, fhi = fhi, pac_method = pac_method, **kwargs)
#     else:
#         xlo, xhi = pac_filter(x_pha, x_amp, fs = fs, flo = flo, fhi = fhi, pac_method = pac_method, **kwargs)
#
# Calculate PAC
#     if pac_method == 'plv':
#         return pac_plv(xlo, xhiamplo)
#     elif pac_method == 'mi_tort':
#         return pac_mi_tort(xlo, xhi)
#     elif pac_method == 'glm':
#         return pac_glm(xlo, xhi)
#     elif pac_method == 'mi_canolty':
#         return pac_mi_canolty(xlo, xhi)
#     elif pac_method == 'otc':
#         _, _, _, pac = otc(x_pha, fs = fs, f0s = np.arange(fhi[0],fhi[1],4))
#         return pac
#     else:
#         raise ValueError('Invalid PAC method')
#
