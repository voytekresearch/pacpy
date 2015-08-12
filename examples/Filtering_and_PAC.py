# -*- coding: utf-8 -*-
"""
This script visualizes the effects of various filter parameters on calculated
beta-high gamma phase-amplitude coupling in an example data set.
"""
#%% Import libraries
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, firwin2, freqz
from matplotlib.widgets import Slider
import math

#%% Function to filter data and calculate PAC

def power_two(n):
    '''
    Calculate the next power of 2 above the number provided
    '''
    return 2**(int(math.log(n, 2))+1)


def fasthilbert(x, axis=-1):
    '''
    Redefinition of scipy.signal.hilbert, which is very slow for some lengths
    of the signal x. This version zero-pads the signal to the next power of 2
    for speed.
    '''
    x = np.array(x)
    N = x.shape[axis]
    N2 = power_two(len(x))
    Xf = np.fft.fft(x, N2, axis=axis)
    h = np.zeros(N2)
    h[0] = 1
    h[1:(N2 + 1) // 2] = 2
        
    x = np.fft.ifft(Xf * h, axis=axis)
    return x[:N]

def rmv_edge(x,win):
    '''
    Remove the edge effects from a signal that you filtered
    '''
    return x[np.int(win):-np.int(win)]

def pac(x,flo, fhi, Fs=1000, bcy = 5, btw = .15, gcy = .5, gtw = .15, gcy_g = False):
    '''
    Calculate PAC using the modulation index method defined in Tort, 2008.
    
    Parameters
    ----------
    x : array
        temporal signal
    flo : array (2x1)
        frequency range for the modulating oscillation (phase frequency)
    fhi : array (2x1)
        frequency range for the modulated oscillation (amplitude frequency)
    Fs : integer
        sampling rate
    bcy : float
        width of filtering window of the low frequency band in terms of the 
        number of cycles of the center frequency of that band
    btw : float (0<x<1)
        transition width of the low frequency band filter as a fraction of
        the frequencies of the passband
    gcy : float
        width of filtering window of the high frequency band in terms of the 
        number of cycles of the center frequency of that band or the beta band
        (depending on gcy_g value)
    gtw : float (0<x<1)
        transition width of the high frequency band filter as a fraction of
        the frequencies of the passband
    gcy_g : boolean
        True : the width of the high frequency band filter is measured as the 
        number of cycles of the center frequency in the high frequency band
        False : the width of the high frequency band filter is measured as the 
        number of cycles of the center frequency in the low frequency band 
  
    Returns
    -------
    amp : array
        Time series of the analytic amplitude of the upper frequency band
    pha : array
        Time series of the analytic phase of the lower frequency band
    pac : numeric
        PAC value
    and now other things
    '''
    bcf = np.mean(flo)
    gcf = np.mean(fhi)
    nyq = Fs/2
    
    
    Ntapsb = np.floor(bcy*Fs/bcf)
    f = [0, (1-btw)*flo[0]/nyq, flo[0]/nyq, flo[1]/nyq, (1+btw)*flo[1]/nyq, 1]
    m = [0,0,1,1,0,0]
    tapsb = firwin2(Ntapsb, f, m)
    xb = filtfilt(tapsb,[1],x)
    # gamma
    if gcy_g:
        Ntapsg = np.floor(gcy*Fs/gcf)
    else:
        Ntapsg = np.floor(gcy*Fs/bcf)
    f = [0, (1-gtw)*fhi[0]/nyq, fhi[0]/nyq, fhi[1]/nyq, (1+gtw)*fhi[1]/nyq, 1]
    m = [0,0,1,1,0,0]
    tapsg = firwin2(Ntapsg, f, m)
    xg = filtfilt(tapsg,[1],x)
    xb = rmv_edge(xb,np.ceil(np.max([Ntapsb,Ntapsg])/2.0))
    xg = rmv_edge(xg,np.ceil(np.max([Ntapsb,Ntapsg])/2.0))
    
    # Calculate frequency responses
    [wb,hb] = freqz(tapsb)
    wb = wb*nyq/np.pi
    [wg,hg] = freqz(tapsg)
    wg = wg*nyq/np.pi
    
    # Calculate high gamma power as function of phase
    amp = np.abs(fasthilbert(xg))
    pha = np.angle(fasthilbert(xb))
    phadeg = np.degrees(pha)
    
    
    # Calculate PAC
    bin_phase_lo = np.arange(-180,180,20)
    binned_meanA = np.zeros(len(bin_phase_lo))
    for b in range(len(bin_phase_lo)):
        phaserange = np.logical_and(phadeg>=bin_phase_lo[b],phadeg<(bin_phase_lo[b]+20))
        binned_meanA[b] = np.mean(amp[phaserange])
        
    p_j = np.zeros(len(bin_phase_lo))
    for b in range(len(bin_phase_lo)):
        p_j[b] = binned_meanA[b]/sum(binned_meanA)
        
    H = -sum(np.multiply(p_j,np.log10(p_j)))
    Hmax = np.log10(18)
    pac_mi = (Hmax-H)/Hmax
        
    # Calculate PAC plv
    amp_beta = filtfilt(tapsb,[1],amp)
    amp_pha = np.angle(fasthilbert(amp_beta))
    pac_plv = np.abs(np.sum(np.exp(1j * (pha - amp_pha)))) / len(pha)
    
    return amp, pha, pac_mi, tapsb, tapsg, wb, hb, wg, hg, bin_phase_lo, binned_meanA, pac_plv

def makeplots(amp, pha, pac_mi, tapsb, tapsg, wb, hb, wg, hg, bin_phase_lo, binned_meanA, pac_plv):
    '''
    Make plots: 
    1. Beta filter impulse response
    2. Beta filter frequency response
    3. Gamma filter impulse response
    4. Gamma filter frequency response
    5. Histogram of gamma power over beta cycle
    '''
    Ntapsb = len(tapsb)
    Ntapsg = len(tapsg)
    
    # Make plots
    plt.subplot(2,3,1)
    plt.cla()
    plt.plot(tapsb)
    plt.xlabel('samples (ms)')
    plt.ylabel('Beta impulse response')
    plt.annotate('Window length: '+np.str(np.int(Ntapsb)),
                xy=(.02, .9),  xycoords='axes fraction')  
    
    plt.subplot(2,3,2)
    plt.cla()
    plt.plot(wb,abs(hb))
    plt.xlim([0,50])
    plt.ylim([0,1])
    plt.xlabel('f (Hz)')
    plt.ylabel('Amplitude (dB)')
    
    plt.subplot(2,3,4)
    plt.cla()
    plt.plot(tapsg)
    plt.xlabel('samples (ms)')
    plt.ylabel('Gamma impulse response')
    plt.annotate('Window length: '+np.str(np.int(Ntapsg)),
                xy=(.02, .9),  xycoords='axes fraction') 
    plt.subplot(2,3,5)
    plt.cla()
    plt.plot(wg,abs(hg))
    plt.ylim([0,1])
    plt.xlim([0,300])
    plt.xlabel('f (Hz)')
    plt.ylabel('Amplitude (dB)')
    
    plt.subplot(1,3,3)
    plt.cla()
    plt.bar(bin_phase_lo,binned_meanA,20)
    plt.ylabel('Mean gamma power')
    plt.xlabel('Beta phase (degrees)')
    plt.xlim([-180,180])
    plt.xticks(np.arange(-180,181,60))
    plt.annotate('\nMI = '+np.str(np.round(pac_mi,4))+'\nPLV = '+np.str(np.round(pac_plv,2)),
                xy=(.02, .9),  xycoords='axes fraction')  
    
    return
    

#%% Look at filters and frequency responses for each subject
# 1. Subject, 2. Beta center frequency, 3. Beta bandwidth, 4. Beta transition width
# 5. Beta # cycles (at center frequency)
# 6. Gamma center frequency, 7. Gamma bandwidth, 8. Gamma transition width
# 9. # beta cycles for the gamma band (.5-1.5) (maybe allow # gamma cycles)

# Plots: 1-4. Beta and Gamma impulse responses and frequency responses
# 5. PAC histogram

# Update functions
def update(val):
    # Get values
    bcf = sbcf.val
    bbw = sbbw.val
    btw = sbtw.val
    bcy = sbcy.val
    gcf = sgcf.val
    gbw = sgbw.val
    gtw = sgtw.val
    gcy = sgcy.val
        
    # Calculate amplitude and phase time series and PAC
    amp, pha, pac_mi, tapsb, tapsg, wb, hb, wg, hg, bin_phase_lo, binned_meanA, pac_plv = pac(
        data,[bcf-bbw,bcf+bbw], [gcf-gbw,gcf+gbw], Fs=1000, bcy = bcy, btw = btw, gcy = gcy, gtw = gtw, gcy_g = False)
    
    makeplots(amp, pha, pac_mi, tapsb, tapsg, wb, hb, wg, hg, bin_phase_lo, binned_meanA, pac_plv)
    
   
#%% Load data
data=np.load('C:/gh/neurosrc/pac/exampledata.npy')
rate = 1000
nyq = rate/2.0

#%% Initial values
bcf0 = 21.5
bbw0 = 8.5
btw0 = .15
bcy0 = 5

gcf0 = 140
gbw0 = 60
gtw0 = .15
gcy0 = .5

# Filter, calculate PAC, and make plots
amp, pha, pac_mi, tapsb, tapsg, wb, hb, wg, hg, bin_phase_lo, binned_meanA, pac_plv = pac(
    data,[bcf0-bbw0,bcf0+bbw0], [gcf0-gbw0,gcf0+gbw0], Fs=1000, bcy = bcy0, btw = btw0, gcy = gcy0, gtw = gtw0, gcy_g = False)
    
fig, ax = plt.subplots(figsize=(20,10))
plt.subplots_adjust(left=0.05, right=0.99, top=0.96, bottom=0.14)

makeplots(amp, pha, pac_mi, tapsb, tapsg, wb, hb, wg, hg, bin_phase_lo, binned_meanA, pac_plv)
   
# Make axes
axcolor = 'lightgoldenrodyellow'
axBcf = plt.axes([0.02, 0.01, 0.2, 0.03], axisbg=axcolor)
axBbw = plt.axes([0.28, 0.01, 0.2, 0.03], axisbg=axcolor)
axBtw = plt.axes([0.53, 0.01, 0.2, 0.03], axisbg=axcolor)
axBcy = plt.axes([0.77, 0.01, 0.2, 0.03], axisbg=axcolor)
axGcf = plt.axes([0.02, 0.05, 0.2, 0.03], axisbg=axcolor)
axGbw = plt.axes([0.28, 0.05, 0.2, 0.03], axisbg=axcolor)
axGtw = plt.axes([0.53, 0.05, 0.2, 0.03], axisbg=axcolor)
axGcy = plt.axes([0.77, 0.05, 0.2, 0.03], axisbg=axcolor)

# Make sliders
sbcf = Slider(axBcf, r'$\beta_{cf}$', 1, 50, bcf0)
sbbw = Slider(axBbw, r'$\beta_{bw}$', 1, 20, bbw0)
sbtw = Slider(axBtw, r'$\beta_{tw}$', 0.01, 0.99, btw0)
sbcy = Slider(axBcy, r'$\beta_{cy}$', 0.5, 10, bcy0)
#
sgcf = Slider(axGcf, r'$\gamma_{cf}$', 50, 200, gcf0)
sgbw = Slider(axGbw, r'$\gamma_{bw}$', 1, 100, gbw0)
sgtw = Slider(axGtw, r'$\gamma_{tw}$', 0.01, 0.99, gtw0)
sgcy = Slider(axGcy, r'$\gamma_{cy}$', 0.1, 10, gcy0)

# Define updates
sbcf.on_changed(update)
sbbw.on_changed(update)
sbtw.on_changed(update)
sbcy.on_changed(update)
sgcf.on_changed(update)
sgbw.on_changed(update)
sgtw.on_changed(update)
sgcy.on_changed(update)