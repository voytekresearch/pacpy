import pytest
import numpy as np
import pacpy
import os
from pacpy.pac import plv, glm, mi_tort, mi_canolty, _plv_postfilt, _mitort_postfilt, _glm_postfilt, _micanolty_postfilt
from pacpy.filt import butterf, firf
from scipy.signal import hilbert

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
    assert glm(data, data, (13,30), (80,200)) == 0.23778487599877976
    
    data=np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    assert glm(data, data, (13,30), (80,200), filterfn = butterf, filter_kwargs = {}) == 0.24695916295388207

def test_mi_tort():
    '''
    Confirm consistency in mi_tort output with both FIR and butterworth filtering
    '''
    # Load data
    data=np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    assert mi_tort(data, data, (13,30), (80,200)) == 0.23778487599877976
    
    data=np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    assert mi_tort(data, data, (13,30), (80,200), filterfn = butterf, filter_kwargs = {}) == 0.24695916295388207

def test_mi_canolty():
    '''
    Confirm consistency in mi_canolty output with both FIR and butterworth filtering
    '''
    # Load data
    data=np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    assert mi_canolty(data, data, (13,30), (80,200)) == 0.23778487599877976
    
    data=np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    assert mi_canolty(data, data, (13,30), (80,200), filterfn = butterf, filter_kwargs = {}) == 0.24695916295388207
    
def test_raiseinputerrors():
    '''
    Confirm that ValueErrors from dumb user input are raised
    '''
    # Load data
    data=np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    
    pytest.raises(ValueError, plv(data, data[:-1], (13,30), (80,200)))
    data2 = data
    data2[-1] = np.nan
    pytest.raises(ValueError, plv(data, data2, (13,30,31), (80,200)))
    pytest.raises(ValueError, plv(data2, data, (13,30), (80,200,201)))
    pytest.raises(ValueError, plv(data, data2, (13,-30), (80,200)))
    pytest.raises(ValueError, plv(data, data2, (13,30), (-80,200)))
    
    pytest.raises(ValueError, mi_tort(data, data2, (13,30), (-80,200), Nbins=1))
    pytest.raises(ValueError, mi_tort(data, data2, (13,30), (-80,200), Nbins=8.8))
    
def test_plvpf():
    '''
    Test that the PLV function outputs 1 when expected
    '''
    # Load data
    data = np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    
    # Test for PLV=1
    dataf = firf(data, (13,30))
    assert _plv_postfilt(dataf, dataf) == 1
    
def test_mitortpf():
    '''
    Test that the Tort MI function outputs 0 and 1 when expected
    '''
    # Load data
    data = np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    
    # Test for MI=0
    dataf = firf(data, (13,30))
    pha = np.angle(hilbert(dataf))
    amp = np.ones(len(pha))
    Nbins = 20
    assert _mitort_postfilt(pha, amp, Nbins) < 10**-10
    
    # Test for MI=1
    T = 1000
    np.random.seed(0)
    pha = np.random.randint(0,Nbins,T)
    pha = (pha - Nbins/2.0) * 2 * np.pi / Nbins
    amp = np.ones(T)*.00001
    amp[pha==0] = 1
    assert _mitort_postfilt(pha, amp, Nbins) > 0.99
    
def test_glmpf():
    '''
    Test that the GLM function outputs 0 and 1 when expected
    '''
    # Load data
    data = np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    
    # Test for GLM = 0
    dataf = firf(data, (13,30))
    pha = np.angle(hilbert(dataf))
    np.random.seed(0)
    amp = np.random.rand(len(pha))
    assert _glm_postfilt(pha, amp) < 10**-4
    
    # Test for GLM = 1
    np.random.seed(0)
    amp = np.sin(pha) + 2*np.cos(pha) + np.random.rand(len(pha))*.001
    assert _glm_postfilt(pha, amp) > 0.99
    
def test_micanoltypf():
    '''
    Test that the Canolty MI function outputs 0 and 1 when expected
    '''
    # Load data
    data = np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    
    # Test for GLM = 0
    dataf = firf(data, (13,30))
    pha = np.angle(hilbert(dataf))
    np.random.seed(0)
    amp = np.random.rand(len(pha))
    print _micanolty_postfilt(pha, amp)# < 10**-4
    
    # Test for GLM = 1
    np.random.seed(0)
    amp = np.cos(pha)
    print _micanolty_postfilt(pha, amp)# > 0.99
    
    