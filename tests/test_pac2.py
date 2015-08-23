import numpy as np
from pacpy.pac import otc, _peaktimes, _chunk_time, comodulogram, pa_series, pa_dist
import os
import pacpy
import pytest

'''
Questions/todo:

'''

def test_otc():
    # Load data
    data=np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    
    # Confirm consistency in result
    t_modsig = (-.5,.5)
    fs = 1000.
    f_hi = (80,200)
    f_step = 4
    pac, tf, a_events, mod_sig = otc(data, f_hi, f_step, fs=fs, t_modsig = t_modsig)
    assert pac# == 220.325631448
    
    # Confirm correct shapes in outputs
    assert np.shape(tf) == ((f_hi[1]-f_hi[0])/f_step,len(data))
    for i in range(len(a_events)):
        assert np.max(a_events[i]) <= len(data)
    assert np.shape(mod_sig) == ((f_hi[1]-f_hi[0])/f_step, len(np.arange(t_modsig[0],t_modsig[1],1/fs)))
    
def test_raiseinputerrors():
    # Load data
    data=np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    
    # Raise correct errors
    with pytest.raises(ValueError) as excinfo:
        otc(data, (80,200), -1)
    assert 'positive number' in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
        otc(data, (80,200), 4, t_modsig = (.5,-.5))
    assert 'Invalid time range' in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
        _peaktimes(data, prc=101)
    assert '0 and 100' in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
        _chunk_time(data, samp_buffer=-1)
    assert 'positive number' in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
        _chunk_time(data, samp_buffer=2.5)
    assert 'integer' in str(excinfo.value)
    
def test_peaktimes():
    # Load data
    data=np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    
    # Confirm functionality
    assert _peaktimes(data[:1000], prc=99) == 344
    assert len(_peaktimes(data[:10000], prc=99)) == 11
    
def test_chunktime():
    assert np.array_equal(_chunk_time([5,6,7,8,10,55,56], samp_buffer = 0), np.array([[ 5,  8], [10, 10], [55, 56]]))
    assert np.array_equal(_chunk_time([5,6,7,8,10,55,56], samp_buffer = 2), np.array([[ 5,  10], [55, 56]]))
    
def test_comod():
    # Load data
    data=np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    p_range = [10,20]
    a_range = [50,150]
    dp = 5
    da = 50
    a = comodulogram(data, data, p_range, a_range, dp, da)
    assert np.round(a[0][0],8) == 0.00286671
    assert np.shape(a) == (len(np.arange(p_range[0],p_range[1],dp)), len(np.arange(a_range[0],a_range[1],da)))
    
def test_paseries():
    # Load data
    data=np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    
    # Confirm returns correct size
    p, a = pa_series(data, data, (13,30), (80,200))
    assert np.shape(p) == np.shape(data)
    assert np.shape(a) == np.shape(data)
    
    # Confirm consistency
    assert np.round(np.mean(a),10) == 11.4397003025
    assert np.round(p[0],11) == 1.57118950513
    
def test_padist():
    # Load data
    data=np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    
    np.random.seed(0)
    Nbins = np.random.randint(2,20)
    pha, amp = pa_series(data, data, (13,30), (80,200))
    dist = pa_dist(pha, amp, Nbins=Nbins)
    assert len(dist) == Nbins
    
    # Confirm consistency
    dist = pa_dist(pha, amp, Nbins=10)
    assert np.round(dist[0],10) == 12.1396071766