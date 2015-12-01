from __future__ import division
import numpy as np
import math
from pacpy.pac import _x_sanity, _range_sanity
from pacpy.filt import firf


def fasthilbert(x, axis=-1):
    """
    Redefinition of scipy.signal.hilbert, which is very slow for some lengths
    of the signal x. This version zero-pads the signal to the next power of 2
    for speed.
    """
    x = np.array(x)
    N = x.shape[axis]
    N2 = 2**(int(math.log(len(x), 2)) + 1)
    Xf = np.fft.fft(x, N2, axis=axis)
    h = np.zeros(N2)
    h[0] = 1
    h[1:(N2 + 1) // 2] = 2

    x = np.fft.ifft(Xf * h, axis=axis)
    return x[:N]


def pha_filtzx(x, f_narrow, fs=1000.,
               Fpeak=True, Ftrough=True, Fzerocross=True,
               filterfn=None, filter_kwargs=None, return_pt=False):
    """

    Calculate the phase time series of 'x' by identifying zerocrossings
    in the filtered signal and updating peaks and troughs to be the argmax
    and argmin in their respective regions.
    Similar to Siapas et al, 2005 except peaks/troughs are based on
    broadband signal

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    f_narrow : (low, high), Hz
        frequency range for narrowband signal of interest, used to find 
        initial guesses of peaks and troughs
    fs : float
        The sampling rate (default = 1000Hz)
    Fpeak : boolean
        if True, peaks times are estimated
    Ftrough : boolean
        if True, troughs times are estimated
    Fzerocross : boolean
        if True, zero crossings are estimated in the same manner 
        as peaks and troughs
    filterfn : function, False
        The filtering function, `filterfn(x, f_range, filter_kwargs)`
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`
    return_pt : boolean
        if True, return the peaks and troughs as arrays in the 2nd and 3rd
        output arguments, respectively

    Returns
    -------
    pha : array-like 1d
        phase time series

    Usage
    -----
    TBD
    """
    # Arg check
    _x_sanity(x)
    _range_sanity(f_narrow)

    if filterfn is None:
        filterfn = firf

    if filter_kwargs is None:
        filter_kwargs = {}

    # Filter in narrow band
    xn = filterfn(x, f_narrow, fs, **filter_kwargs)
    # Initialize phase array
    L = len(xn)
    pha = np.empty(L)
    pha[:] = np.NAN

    # Find zero crosses
    def fzerofall(data):
        pos = data > 0
        return (pos[:-1] & ~pos[1:]).nonzero()[0]

    def fzerorise(data):
        pos = data < 0
        return (pos[:-1] & ~pos[1:]).nonzero()[0]

    zeroriseN = fzerorise(xn)
    zerofallN = fzerofall(xn)

    # Calculate # peaks and troughs
    if zeroriseN[-1] > zerofallN[-1]:
        P = len(zeroriseN) - 1
        T = len(zerofallN)
    else:
        P = len(zeroriseN)
        T = len(zerofallN) - 1

    # Calculate peak samples
    peaksB = np.zeros(P, dtype=int)
    for p in range(P):
        # Calculate the sample range between the most recent zero rise
        # and the next zero fall
        try:
            mrzerorise = zeroriseN[p]
            nfzerofall = zerofallN[zerofallN > mrzerorise][0]
            peaksB[p] = np.argmax(x[mrzerorise:nfzerofall]) + mrzerorise
        except:
            import pdb
            pdb.set_trace()

    # Calculate trough samples
    troughsB = np.zeros(T, dtype=int)
    for tr in range(T):
        # Calculate the sample range between the most recent zero fall
        # and the next zero rise
        mrzerofall = zerofallN[tr]
        nfzerorise = zeroriseN[zeroriseN > mrzerofall][0]
        troughsB[tr] = np.argmin(x[mrzerofall:nfzerorise]) + mrzerofall

    # Label peaks
    if Fpeak:
        try:
            pha[peaksB] = 0
        except:
            import pdb
            pdb.set_trace()

    # Label troughs
    if Ftrough:
        pha[troughsB] = -np.pi

    # Label zerocross
    if Fzerocross:
        pha[zeroriseN] = -np.pi / 2.
        pha[zerofallN] = np.pi / 2.

    # Interpolate to find all phases
    marks = np.logical_not(np.isnan(pha))
    t = np.arange(L)
    marksT = t[marks]
    M = len(marksT)
    for m in range(M - 1):
        idx1 = marksT[m]
        idx2 = marksT[m + 1]

        val1 = pha[idx1]
        val2 = pha[idx2]
        if val2 <= val1:
            val2 = val2 + 2 * np.pi

        phatemp = np.linspace(val1, val2, idx2 - idx1 + 1)
        pha[idx1:idx2] = phatemp[:-1]

    # Interpolate the boundaries with the same rate of change as the adjacent
    # sections
    idx = np.where(np.logical_not(np.isnan(pha)))[0][0]
    val = pha[idx]
    dval = pha[idx + 1] - val
    startval = val - dval * idx
    # .5 for nonambiguity in arange length
    pha[:idx] = np.arange(startval, val - dval * .5, dval)

    idx = np.where(np.logical_not(np.isnan(pha)))[0][-1]
    val = pha[idx]
    dval = val - pha[idx - 1]
    dval = np.angle(np.exp(1j * dval))  # Trestrict dval to between -pi and pi
    # .5 for nonambiguity in arange length
    endval = val + dval * (len(pha) - idx - .5)
    pha[idx:] = np.arange(val, endval, dval)

    # Restrict phase between -pi and pi
    pha = np.angle(np.exp(1j * pha))

    if return_pt:
        return pha, peaksB, troughsB
    else:
        return pha