from __future__ import division
import numpy as np
from scipy.signal import argrelextrema
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
    N2 = 2**(int(math.log(len(x), 2))+1)
    Xf = np.fft.fft(x, N2, axis=axis)
    h = np.zeros(N2)
    h[0] = 1
    h[1:(N2 + 1) // 2] = 2
        
    x = np.fft.ifft(Xf * h, axis=axis)
    return x[:N]
    
    
def pha_waveform(x, f_narrow,
                Fpeak = True, Ftrough = True, Fzerocross = True, 
                fs = 1000., filterfn=None, filter_kwargs=None):
    """
    IN DEVELOPMENT
    
    WAAAAAAAAAAAAIT. I'm not using the peaks/troughs from the narrow signal;
    I'm only using the zero crossings. The only time I need the peaks/troughs
    is at the edges of the signal.
    
    Calculate the phase time series of 'x' by using the method outline in
    Trimper et al. 2014.
    
    Parameters
    ----------
    x : array-like 1d
        voltage time series
    f_narrow : (low, high), Hz
        frequency range for narrowband signal of interest, used to find initial
        guesses of peaks and troughs
    Fpeak : boolean
        if True, peaks times are estimated
    Ftrough : boolean
        if True, troughs times are estimated
    Fzerocross : boolean
        if True, zero crossings are estimated in the same manner as peaks and
        troughs
    fs : float
        The sampling rate (default = 1000Hz)
    filterfn : function, False
        The filtering function, `filterfn(x, f_range, filter_kwargs)`
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`
        
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
    
    # Find extrema
    def fzerofall(data):
        pos = data > 0
        return (pos[:-1] & ~pos[1:]).nonzero()[0]
        
    def fzerorise(data):
        pos = data < 0
        return (pos[:-1] & ~pos[1:]).nonzero()[0]
        
    zeroriseN = fzerorise(xn)
    zerofallN = fzerofall(xn)
    
    # OLD METHOD WITH FINDINGS NARROW PEAKS/TROUGHS
#    peaksN = argrelextrema(xn, np.greater)
#    troughsN = argrelextrema(xn, np.less)
#    # Delete a peak if the most recent trough is after the most recent zerorise
#    # Also delete that trough
#    P = len(peaksN)
#    peaksBAD = np.zeros(0)
#    troughsBAD = np.zeros(0)
#    for p in range(P):
#        peaksamp = peaksN[p]
#        rtrough = troughsN[troughsN < peaksamp]
#        rzerorise = zeroriseN[zeroriseN < peaksamp]
#        
#        if np.logical_and(len(rtrough)>0, len(rzerorise>0)):
#            mrtrough = rtrough[-1]
#            mrzerorise= rzerorise[-1]
#            
#            if mrtrough > mrzerorise:
#                peaksBAD = np.append(peaksBAD,p)
#                troughsBAD = np.append(troughsBAD,len(rtrough)-1)
#        
#    peaksN = np.delete(peaksN,peaksBAD)
#    troughsN = np.delete(troughsN,troughsBAD)
#    
#    # Update peaks, troughs, and zerocross with raw signal
#    P = len(peaksN)
#    peaksB = np.zeros(P)
#    for p in range(P):
#        # Calculate the sample range between the most recent zero rise
#        # and the next zero fall
#        peaksamp = peaksN[p]
#        rzerorise = zeroriseN[zeroriseN < peaksamp]
#        fzerofall = zerofallN[zerofallN > peaksamp]
#        
#        if len(fzerofall)>0:
#            nfzerofall = fzerofall[0]
#        else:
#            nfzerofall = len(x)
#            
#        if len(rzerorise>0):
#            mrzerorise= rzerorise[-1]
#        else:
#            mrzerorise = 0
#        
#        peaksB[p] = np.argmax(x[mrzerorise:nfzerofall])
        
    # Calculate # peaks and troughs
    if np.logical_xor(zeroriseN[0] < zerofallN[0], zeroriseN[-1] > zerofallN[-1]):
        P = len(zeroriseN)
        T = len(zerofallN) - 1
    else:
        P = len(zeroriseN) - 1
        T = len(zerofallN)
        
    # Calculate peak samples
    peaksB = np.zeros(P)
    for p in range(P):
        # Calculate the sample range between the most recent zero rise
        # and the next zero fall
        mrzerorise = zeroriseN[p]
        nfzerofall = zerofallN[zerofallN > mrzerorise][0]
        
        # If there is an error here, then I miscalculated # Peaks and # troughs
#        if len(fzerofall)>0:
#            nfzerofall = fzerofall[0]
#        else:
#            nfzerofall = len(x)
        peaksB[p] = np.argmax(x[mrzerorise:nfzerofall]) + mrzerorise
        
    # Calculate trough samples
    troughsB = np.zeros(T)
    for tr in range(T):
        # Calculate the sample range between the most recent zero fall
        # and the next zero rise
        mrzerofall = zerofallN[tr]
        try:
            nfzerorise = zeroriseN[zeroriseN > mrzerofall][0]
        except:
            import pdb
            pdb.set_trace()
        
        # Calculate trough time
        troughsB[tr] = np.argmin(x[mrzerofall:nfzerorise]) + mrzerofall
        
    # Recalculate zerocrossings
    R = len(zeroriseN)
    zeroriseB = np.zeros(R)
    for r in range(R):
        # Calculate the sample range between the most recent trough and next peak
        zrsamp = zeroriseN[r]
        rtrough = troughsB[troughsB < zrsamp]
        fpeak = peaksB[peaksB > zrsamp]
        
        if len(fpeak)>0:
            nfpeak = fpeak[0]
        else:
            nfpeak = len(x)
            
        if len(rtrough>0):
            mrtrough= rtrough[-1]
        else:
            mrtrough = 0
        
        Tzerocross = fzerorise(x[mrtrough:nfpeak])
        # If there is more than 1 zerocross, then take the mean of first and last
        try:
            Tzerocross = np.mean(Tzerocross[[0,-1]])
        except:
            
            import pdb
            pdb.set_trace()
        zeroriseB[p] = Tzerocross + mrtrough
        
    F = len(zerofallN)
    zerofallB = np.zeros(F)
    for f in range(F):
        # Calculate the sample range between the most recent trough and next peak
        zfsamp = zerofallN[f]
        rpeak = peaksB[peaksB < zfsamp]
        ftrough = troughsB[troughsB > zfsamp]
        
        if len(rpeak)>0:
            mrpeak = rpeak[0]
        else:
            mrpeak = len(x)
            
        if len(ftrough>0):
            nftrough= ftrough[-1]
        else:
            nftrough = 0
        
        Tzerocross = fzerofall(x[mrpeak:nftrough])
        try:
            # If there is more than 1 zerocross, then take the mean of first and last
            Tzerocross = np.mean(Tzerocross[[0,-1]])
        except:
            import pdb
            pdb.set_trace()
        zerofallB[p] = Tzerocross + mrpeak
    
        
    # Label peaks
    if Fpeak:
        pha[peaksB] = 0
        
    # Label troughs
    if Ftrough:
        pha[troughsB] = -np.pi
        
    # Label zerocross
    if Fzerocross:
        pha[zeroriseB] = -np.pi/2.
        pha[zerofallB] = np.pi/2.
    
    # Interpolate to find all phases
    marks = np.logical_not(np.isnan(pha))
    t = np.arange(L)
    marksT = t[marks]
    M = len(marksT)
    for m in range(M-1):
        idx1 = marksT[m]
        idx2 = marksT[m+1]
        
        val1 = pha[idx1]
        val2 = pha[idx2]
        if val2 <= val1:
            val2 = val2 + 2*np.pi
            
        phatemp = np.linspace(val1,val2,idx2-idx1+1)
        pha[idx1:idx2] = phatemp[:-1]
    
    # Interpolate the boundaries with the same rate of change as the adjacent sections
    idx = np.where(np.logical_not(np.isnan(pha)))[0][0]
    val = pha[idx]
    dval = pha[idx+1] - val
    startval = val - dval*idx
    pha[:idx] = np.arange(startval,val-dval*.5,dval) #.5 for nonambiguity in arange length
    
    idx = np.where(np.logical_not(np.isnan(pha)))[0][-1]
    val = pha[idx]
    dval = val - pha[idx-1]
    dval = np.angle(np.exp(1j*dval)) # Trestrict dval to between -pi and pi
    endval = val + dval*(len(pha) - idx - .5) #.5 for nonambiguity in arange length
    pha[idx:] = np.arange(val,endval,dval)
                       
    # Restrict phase between -pi and pi
    pha = np.angle(np.exp(1j*pha))
    return pha
    
    
def pha_filtzx(x, f_narrow, fs = 1000., 
               Fpeak = True, Ftrough = True, Fzerocross = True, 
               filterfn=None, filter_kwargs=None, return_pt = False):
    """
    
    Calculate the phase time series of 'x' by identifying zerocrossings
    in the filtered signal and updating peaks and troughs to be the argmax
    and argmin in their respective regions.
    Similar to Siapas et al, 2005 except peaks/troughs are based on broadband
    signal
    
    Parameters
    ----------
    x : array-like 1d
        voltage time series
    f_narrow : (low, high), Hz
        frequency range for narrowband signal of interest, used to find initial
        guesses of peaks and troughs
    fs : float
        The sampling rate (default = 1000Hz)
    Fpeak : boolean
        if True, peaks times are estimated
    Ftrough : boolean
        if True, troughs times are estimated
    Fzerocross : boolean
        if True, zero crossings are estimated in the same manner as peaks and
        troughs
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
    peaksB = np.zeros(P,dtype=int)
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
    troughsB = np.zeros(T,dtype=int)
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
        pha[zeroriseN] = -np.pi/2.
        pha[zerofallN] = np.pi/2.
    
    # Interpolate to find all phases
    marks = np.logical_not(np.isnan(pha))
    t = np.arange(L)
    marksT = t[marks]
    M = len(marksT)
    for m in range(M-1):
        idx1 = marksT[m]
        idx2 = marksT[m+1]
        
        val1 = pha[idx1]
        val2 = pha[idx2]
        if val2 <= val1:
            val2 = val2 + 2*np.pi
            
        phatemp = np.linspace(val1,val2,idx2-idx1+1)
        pha[idx1:idx2] = phatemp[:-1]
    
    # Interpolate the boundaries with the same rate of change as the adjacent sections
    idx = np.where(np.logical_not(np.isnan(pha)))[0][0]
    val = pha[idx]
    dval = pha[idx+1] - val
    startval = val - dval*idx
    pha[:idx] = np.arange(startval,val-dval*.5,dval) #.5 for nonambiguity in arange length
    
    idx = np.where(np.logical_not(np.isnan(pha)))[0][-1]
    val = pha[idx]
    dval = val - pha[idx-1]
    dval = np.angle(np.exp(1j*dval)) # Trestrict dval to between -pi and pi
    endval = val + dval*(len(pha) - idx - .5) #.5 for nonambiguity in arange length
    pha[idx:] = np.arange(val,endval,dval)
                       
    # Restrict phase between -pi and pi
    pha = np.angle(np.exp(1j*pha))
    
    if return_pt:
        return pha, peaksB, troughsB
    else:
        return pha
    
    
    
def em_lfpow():
    """
    EXPERT MODE TOOL:
    only use the time points of highest low frequency power
    """
    return 0
    
def removeedge(x, samp):
    '''
    Trim the array x on both sides by the number of samples in 'samp' in order
    to remove effects of filtering edge artifact
    '''
    return x[samp:-samp]