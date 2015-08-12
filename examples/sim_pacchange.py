# -*- coding: utf-8 -*-
"""
Calculate PAC changes for different changes in spike statistics
Change 1 ('chg1') : Increase phase-independent firing rate
Change 2 ('chg2') : Decrease the fraction of spikes that are phase-dependent
Change 3 ('chg3') : Increase all firing rate
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from NeuroTools import stgen
from pacpy.pac import pac, rmv_edgeart, phaseamp_series, calcHGPdist

#%% Simulate spike trains for LFP
np.random.seed(1)
stg = stgen.StGen()
mod_frac = .6 #Phase-dependent modulation is this fraction of spikes
mod_f_mu = 20.0 #Mean modulation frequency
mod_f_std = 3.0 #StDev of modulation frequency
firing_rate = 5.0 #Firing rate for each neuron in Hz
nNeu = 100.0
dt = .001
Fs = 1/dt

Nepochs = 60
Tepochs = .5

factor_chg1 = 2.0
factor_chg2 = .2
factor_chg3 = 1.5

for e in range(Nepochs):
    # Create the modulating oscillation
    mod_freq = mod_f_mu + mod_f_std*np.random.randn()
    mod_phase = (np.arange(0,Tepochs,dt) % (1/mod_freq)) * 2*np.pi*mod_freq
    mod_phase = (mod_phase + 2*np.pi*np.random.rand()) % (2*np.pi)
    
    # Calculate firing rate time series
    ctrl_dep = np.sin(mod_phase) + 1
    ctrl_dep = ctrl_dep * firing_rate * mod_frac
    ctrl_indep = firing_rate*(1-mod_frac) * np.ones(Tepochs/dt)
    ctrl_r = ctrl_dep + ctrl_indep 
    
    # Simulate neuronal firing for control population
    if e == 0:
        spikes_ctrl = stg.inh_poisson_generator(ctrl_r*nNeu/dt,np.arange(0,Tepochs,dt),Tepochs,array=True)
    else:
        spikes_ctrl = np.hstack([spikes_ctrl,stg.inh_poisson_generator(ctrl_r*nNeu/dt,np.arange(0,Tepochs,dt),Tepochs,array=True)+Tepochs*e])
        
    # Calculate firing rate time series for change 1 and simulate firing
    chg1_dep = ctrl_dep
    chg1_indep = ctrl_indep + factor_chg1*np.ones(Tepochs/dt)
    chg1_r = chg1_dep + chg1_indep
    
    if e == 0:
        spikes_chg1 = stg.inh_poisson_generator(chg1_r*nNeu/dt,np.arange(0,Tepochs,dt),Tepochs,array=True)
    else:
        spikes_chg1 = np.hstack([spikes_chg1,stg.inh_poisson_generator(chg1_r*nNeu/dt,np.arange(0,Tepochs,dt),Tepochs,array=True)+Tepochs*e])
        
    # Calculate firing rate time series for change 2 and simulate firing
    chg2_dep = np.sin(mod_phase) + 1
    chg2_dep = chg2_dep * firing_rate * factor_chg2
    chg2_indep = firing_rate*(1-factor_chg2) * np.ones(Tepochs/dt)
    chg2_r = chg2_dep + chg2_indep 
    
    if e == 0:
        spikes_chg2 = stg.inh_poisson_generator(chg2_r*nNeu/dt,np.arange(0,Tepochs,dt),Tepochs,array=True)
    else:
        spikes_chg2 = np.hstack([spikes_chg2,stg.inh_poisson_generator(chg2_r*nNeu/dt,np.arange(0,Tepochs,dt),Tepochs,array=True)+Tepochs*e])
        
    # Simulate firing for change 3
    if e == 0:
        spikes_chg3 = stg.inh_poisson_generator(factor_chg3*ctrl_r*nNeu/dt,np.arange(0,Tepochs,dt),Tepochs,array=True)
    else:
        spikes_chg3 = np.hstack([spikes_chg3,stg.inh_poisson_generator(factor_chg3*ctrl_r*nNeu/dt,np.arange(0,Tepochs,dt),Tepochs,array=True)+Tepochs*e])

#%% Visualize spike trains
t_plot = np.arange(0,Tepochs,dt)
neus_ctrl = np.random.randint(0,nNeu,len(spikes_ctrl))
neus_chg1 = np.random.randint(0,nNeu,len(spikes_chg1))
neus_chg2 = np.random.randint(0,nNeu,len(spikes_chg2))
neus_chg3 = np.random.randint(0,nNeu,len(spikes_chg3))

plt.figure()

plt.subplot(4,2,1)
plt.plot(t_plot,ctrl_indep,label='phase-independent')
plt.plot(t_plot,ctrl_dep,label='phase-dependent')
plt.plot(t_plot,ctrl_r,label='total')
plt.ylabel('Firing rate; DBS ctrl')
plt.legend()

plt.subplot(4,2,2)
plt.plot(spikes_ctrl[spikes_ctrl<.5],neus_ctrl[spikes_ctrl<.5],'.',ms=5)
plt.ylabel('Neuron')

plt.subplot(4,2,3)
plt.plot(t_plot,chg1_indep,label='phase-independent')
plt.plot(t_plot,chg1_dep,label='phase-dependent')
plt.plot(t_plot,chg1_r,label='total')
plt.ylabel('Firing rate; vertical shift')

plt.subplot(4,2,4)
plt.plot(spikes_chg1[spikes_chg1<.5],neus_chg1[spikes_chg1<.5],'.',ms=5)
plt.ylabel('Neuron')

plt.subplot(4,2,5)
plt.plot(t_plot,chg2_indep,label='phase-independent')
plt.plot(t_plot,chg2_dep,label='phase-dependent')
plt.plot(t_plot,chg2_r,label='total')
plt.ylabel('Firing rate; flatten')
plt.xlabel('Time (ms)')

plt.subplot(4,2,6)
plt.plot(spikes_chg2[spikes_chg2<.5],neus_chg2[spikes_chg2<.5],'.',ms=5)
plt.ylabel('Neuron')

plt.subplot(4,2,7)
plt.plot(t_plot,factor_chg3*ctrl_indep,label='phase-independent')
plt.plot(t_plot,factor_chg3*ctrl_dep,label='phase-dependent')
plt.plot(t_plot,factor_chg3*ctrl_r,label='total')
plt.ylabel('Firing rate; vertical stretch')
plt.xlabel('Time (ms)')

plt.subplot(4,2,8)
plt.plot(spikes_chg3[spikes_chg3<.5],neus_chg3[spikes_chg3<.5],'.',ms=5)
plt.ylabel('Neuron')
plt.xlabel('Time (ms)')

#%% Calculate LFP

# Define alpha function
t_alpha = np.arange(100) # Alpha time course (ms)
tau_rise = 3
tau_decay = 40
#tau_rise = 0.5
#tau_decay = 1.5
gmax = 1
alpha = gmax * (np.exp(-t_alpha/tau_decay) - np.exp(-t_alpha/tau_rise))

# Convert list of spike times to a temporal array 
splist_ctrl, t_ctrl = np.histogram(spikes_ctrl,np.arange(0,Tepochs*Nepochs,dt))
splist_chg1, t_chg1 = np.histogram(spikes_chg1,np.arange(0,Tepochs*Nepochs,dt))
splist_chg2, t_chg2 = np.histogram(spikes_chg2,np.arange(0,Tepochs*Nepochs,dt))
splist_chg3, t_chg3 = np.histogram(spikes_chg3,np.arange(0,Tepochs*Nepochs,dt))

# Convolve spike train with alpha
lfp_ctrl = np.convolve(splist_ctrl,alpha,mode='same')
lfp_chg1 = np.convolve(splist_chg1,alpha,mode='same')
lfp_chg2 = np.convolve(splist_chg2,alpha,mode='same')
lfp_chg3 = np.convolve(splist_chg3,alpha,mode='same')

# Plot LFP
plt.figure()
plt.subplot(4,1,1)
plt.plot(t_ctrl[:5000],lfp_ctrl[:5000])
plt.ylabel('Control')
plt.subplot(4,1,2)
plt.plot(t_chg1[:5000],lfp_chg1[:5000])
plt.ylabel('Phase-independent rate increase')
plt.subplot(4,1,3)
plt.plot(t_chg2[:5000],lfp_chg2[:5000])
plt.ylabel('Phase-dependent rate redistribution')
plt.subplot(4,1,4)
plt.plot(t_chg3[:5000],lfp_chg3[:5000])
plt.ylabel('Proportional rate increase')

#%% Calculate pac
Fs = 1/dt
fpachi = [80, 200]
cf = mod_f_mu
beta_hw = 6
pac_method = 'mi_tort'
kwargs = {'w_lo' : 7}
indi_beta = (cf - beta_hw, cf + beta_hw)

pac_ctrl = pac(lfp_ctrl, Fs = Fs,
                flo = indi_beta, fhi = fpachi,
                pac_method = pac_method, **kwargs)
pac_chg1 = pac(lfp_chg1, Fs = Fs,
                flo = indi_beta, fhi = fpachi,
                pac_method = pac_method, **kwargs)
pac_chg2 = pac(lfp_chg2, Fs = Fs,
                flo = indi_beta, fhi = fpachi,
                pac_method = pac_method, **kwargs)
pac_chg3 = pac(lfp_chg3, Fs = Fs,
                flo = indi_beta, fhi = fpachi,
                pac_method = pac_method, **kwargs)

print 'PAC'
print pac_ctrl
print pac_chg1
print pac_chg2
print pac_chg3

#%% Calculate HGP distribution over cycle

# Calculate phase and amplitude time series
betaphase_ctrl, gammapower_ctrl = phaseamp_series(lfp_ctrl, Fs = Fs,
        flo = indi_beta, fhi = fpachi, pac_method = pac_method, **kwargs)
betaphase_chg1, gammapower_chg1 = phaseamp_series(lfp_chg1, Fs = Fs,
        flo = indi_beta, fhi = fpachi, pac_method = pac_method, **kwargs)
betaphase_chg2, gammapower_chg2 = phaseamp_series(lfp_chg2, Fs = Fs,
        flo = indi_beta, fhi = fpachi, pac_method = pac_method, **kwargs)
betaphase_chg3, gammapower_chg3 = phaseamp_series(lfp_chg3, Fs = Fs,
        flo = indi_beta, fhi = fpachi, pac_method = pac_method, **kwargs)
   
# Calculate HGP distributions
nPhaseBins = 10
hgpdist_ctrl = calcHGPdist(betaphase_ctrl, gammapower_ctrl, nPhaseBins=nPhaseBins)
hgpdist_chg1 = calcHGPdist(betaphase_chg1, gammapower_chg1, nPhaseBins=nPhaseBins)
hgpdist_chg2 = calcHGPdist(betaphase_chg2, gammapower_chg2, nPhaseBins=nPhaseBins)
hgpdist_chg3 = calcHGPdist(betaphase_chg3, gammapower_chg3, nPhaseBins=nPhaseBins)
boundsPhaseBins = np.linspace(-np.pi,np.pi,nPhaseBins+1)

# Plot HGP distributions
plt.figure()
plt.plot(boundsPhaseBins,np.hstack([hgpdist_ctrl,hgpdist_ctrl[0]]),marker='.', label='ctrl')
plt.plot(boundsPhaseBins,np.hstack([hgpdist_chg1,hgpdist_chg1[0]]),marker='.', label='chg1')
plt.plot(boundsPhaseBins,np.hstack([hgpdist_chg2,hgpdist_chg2[0]]),marker='.', label='chg2')
plt.plot(boundsPhaseBins,np.hstack([hgpdist_chg3,hgpdist_chg3[0]]),marker='.', label='chg3')
plt.xlabel('Beta phase')
plt.ylabel('Mean high gamma amplitude')
plt.legend(loc='best')
plt.xlim([-np.pi-.1,np.pi+.1])
plt.xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],[r'$-\pi$',r'$-\pi/2$','0',r'$\pi/2$',r'$\pi$'])