import pytest
import numpy as np
import pacpy
import os
from pacpy.pac import plv, glm, mi_tort, mi_canolty, ozkurt
from pacpy.filt import butterf, firf
from scipy.signal import hilbert
import copy

'''
Questions/todo:

'''

def test_plv():
    '''
    Confirm consistency in plv output with both FIR and butterworth filtering
    '''
    # Load data
    data=np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    assert plv(data, data, (13,30), (80,200)) == 0.23778487599877976
    
    data=np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    assert plv(data, data, (13,30), (80,200), filterfn = butterf, filter_kwargs = {}) == 0.24695916295388207

def test_glm():
    '''
    Confirm consistency in glm output with both FIR and butterworth filtering
    '''
    # Load data
    data=np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    assert glm(data, data, (13,30), (80,200)) == 0.031906460221
    
    data=np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    assert glm(data, data, (13,30), (80,200), filterfn = butterf, filter_kwargs = {}) == 0.0347586202162

def test_mi_tort():
    '''
    Confirm consistency in mi_tort output with both FIR and butterworth filtering
    '''
    # Load data
    data=np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    assert mi_tort(data, data, (13,30), (80,200)) == 0.00365835870397
    
    data=np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    assert mi_tort(data, data, (13,30), (80,200), filterfn = butterf, filter_kwargs = {}) == 0.00429228117117

def test_mi_canolty():
    '''
    Confirm consistency in mi_canolty output with both FIR and butterworth filtering
    '''
    # Load data
    data=np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    assert mi_canolty(data, data, (13,30), (80,200)) == 1.10063296657
    
    data=np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    assert mi_canolty(data, data, (13,30), (80,200), filterfn = butterf, filter_kwargs = {}) == 1.14299920997
    
def test_ozkurt():
    '''
    Confirm consistency in ozkurt PAC output with both FIR and butterworth filtering
    '''
    # Load data
    data=np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    assert np.round(ozkurt(data, data, (13,30), (80,200)),13) == 0.0754764820714
    
    data=np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    assert np.round(ozkurt(data, data, (13,30), (80,200), filterfn = butterf, filter_kwargs = {}),13) == 0.0755537874324
    
def test_raiseinputerrors():
    '''
    Confirm that ValueErrors from dumb user input are raised
    '''
    # Load data
    data=np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    data2 = copy.copy(data)
    data2[-1] = np.nan
    
    with pytest.raises(ValueError) as excinfo:
        plv(data, data[:-1], (13,30), (80,200))
    assert 'same length' in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
        plv(data, data2, (13,30), (80,200))
    assert 'NaNs' in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
        plv(data, data, (13,30,31), (80,200))
    assert 'two elements' in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
        plv(data, data, (13,30), (80,200,201))
    assert 'two elements' in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
        plv(data, data, (-13,30), (80,200))
    assert 'must be > 0' in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
        plv(data, data, (13,30), (-80,200))
    assert 'must be > 0' in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
        mi_tort(data, data, (13,30), (80,200), Nbins=1)
    assert 'integer >1' in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
        mi_tort(data, data, (13,30), (80,200), Nbins=8.8)
    assert 'integer >1' in str(excinfo.value)
    

def genPAC1(phabias = .5, flo = 5, fhi = 100, glm_bias = False):
    '''
    Generate two signals that have very high PAC
    phase oscillation = 5Hz
    amplitude oscillation = 100Hz
    '''
    dt = .001
    T = 10
    t = np.arange(0,T,dt)
    
    lo = np.sin(t*2*np.pi*flo)
    hi = np.sin(t*2*np.pi*fhi)
    
    if glm_bias:
        hi = hi * (lo+1)
    else:
        pha = np.angle(hilbert(lo))
        hi[pha > -np.pi + phabias] = 0
        
    return lo, hi
    
    
def genPAC0(flo = 5, fhi = 100):
    '''
    Generate two signals that have very low PAC
    phase oscillation = 5Hz
    amplitude oscillation = 100Hz
    '''
    dt = .001
    T = 10
    t = np.arange(0,T,dt)
    
    lo = np.sin(t*2*np.pi*flo)
    hi = np.sin(t*2*np.pi*fhi)
    return lo, hi
    

def ident(x, f, fs):
    return x
        
        
def test_plvpf():
    '''
    Test that the PLV function outputs close to 0 and 1 when expected
    '''
    lo, hi = genPAC1()
    assert plv(lo, hi, (4,6), (90,110)) > 0.99
    
    lo, hi = genPAC0()
    assert plv(lo, hi, (4,6), (90,110)) < 0.05
    
    
def test_mitortpf():
    '''
    Test that the Tort MI function outputs close to 0 and 1 when expected
    '''
    lo, hi = genPAC1(phabias = .2, fhi = 300)
    assert mi_tort(lo, hi, (4,6), (100, 400)) > 0.8
    
    lo, hi = genPAC0()
    assert mi_tort(lo, hi, (4,6), (90,110)) < 10**-5
    
    
def test_glmpf():
    '''
    Test that the GLM function outputs close to 0 and 1 when expected
    '''
    lo, hi = genPAC1(glm_bias = True)
    assert glm(lo, hi, (4,6), (90,110)) > 0.99
    
    lo, hi = genPAC0()
    assert glm(lo, hi, (4,6), (90,110)) < 0.01
    
def test_micanoltypf():
    '''
    Test that the Canolty MI function outputs close to 0 and 1 when expected
    '''
    lo, hi = genPAC1(phabias = .2, fhi = 300)
    hif = firf(hi, (100,400))
    amp = np.abs(hilbert(hif))
    assert mi_canolty(lo, hi, (4,6), (100, 400))/np.mean(amp) > 0.99
    
    lo, hi = genPAC0()
    assert mi_canolty(lo, hi, (4,6), (90,110)) < 0.001
    
def test_ozkurtpf():
    '''
    Test that the Ozkurt PAC function outputs close to 0 and 1 when expected
    '''
    lo, hi = genPAC1(phabias = .2, fhi = 300)
    hif = firf(hi, (100,400))
    amp = np.abs(hilbert(hif))
    weight = (np.sqrt(len(amp)) * np.sqrt(np.sum(amp**2))) / np.sum(amp)
    assert ozkurt(lo, hi, (4,6), (100, 400)) * weight > 0.99
    
    lo, hi = genPAC0()
    assert ozkurt(lo, hi, (4,6), (90,110)) < 0.001