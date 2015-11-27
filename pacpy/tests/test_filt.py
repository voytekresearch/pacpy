import numpy as np
import pacpy
import os
from pacpy.filt import butterf, firfls, morletf, rmv_edgeart, morletT, firf


def test_firfls():
    """
    Confirm consistency in FIR filtering
    """
    # Load data
    data = np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    assert np.allclose(
        np.sum(np.abs(firfls(data, (13, 30)))), 6421935.78673, atol=10 ** -5)
    assert len(firfls(data, (13, 30))) == len(data)


def test_butterf():
    """
    Confirm consistency in butterworth filtering
    """
    # Load data
    data = np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    assert np.allclose(
        np.sum(np.abs(butterf(data, (13, 30)))), 6339982.78280, atol=10 ** -5)
    assert len(butterf(data, (13, 30))) == len(data)


def test_morletf():
    """
    Confirm consistency in morlet wavelet filtering
    """
    # Load data
    data = np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    assert np.allclose(
        np.sum(np.abs(morletf(data, 21.5))), 40661855.060118973, atol=10 ** -5)
    assert len(morletf(data, 21.5)) == len(data)

def test_morletT():
    """
    Confirm that function output size is consistent with inputs
    """
    # Load data
    data = np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    f0s = np.arange(80, 4, 150)
    tf = morletT(data, f0s)
    assert np.shape(tf) == (len(f0s), len(data))