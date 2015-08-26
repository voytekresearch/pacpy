import numpy as np
import pacpy
import os
from pacpy.filt import butterf, firf, morletf, rmv_edgeart, morletT

def test_firf():
    """
    Confirm consistency in FIR filtering
    """
    # Load data
    data=np.load(os.path.dirname(pacpy.__file__) + '/test/exampledata.npy')
    assert np.allclose(np.sum(np.abs(firf(data, (13,30)))), 6421935.78673, atol = 10**-5)
    assert len(firf(data, (13,30))) == len(data)
    
def test_butterf():
    """
    Confirm consistency in butterworth filtering
    """
    # Load data
    data=np.load(os.path.dirname(pacpy.__file__) + '/test/exampledata.npy')
    assert np.allclose(np.sum(np.abs(butterf(data, (13,30)))), 6339982.78280, atol = 10**-5)
    assert len(butterf(data, (13,30))) == len(data)
    
def test_morletf():
    """
    Confirm consistency in morlet wavelet filtering
    """
    # Load data
    data=np.load(os.path.dirname(pacpy.__file__) + '/test/exampledata.npy')
    assert np.allclose(np.sum(np.abs(morletf(data, 21.5))), 40661855.060118973, atol = 10**-5)
    assert len(morletf(data, 21.5)) == len(data)
    
def test_rmvedge():
    """
    Confirm that function is removing edges appropriately
    """
    # Load data
    data=np.load(os.path.dirname(pacpy.__file__) + '/test/exampledata.npy')
    fs = 1000
    cf = 20
    w = 7
    win = np.floor((w * fs / cf) / 2.0)
    assert len(rmv_edgeart(data, w, cf, fs)) == len(data) - 2*win
    
def test_morletT():
    """
    Confirm that function output size is consistent with inputs
    """
    # Load data
    data=np.load(os.path.dirname(pacpy.__file__) + '/test/exampledata.npy')
    f0s = np.arange(80,4,150)
    tf = morletT(data, f0s)
    assert np.shape(tf) == (len(f0s),len(data))