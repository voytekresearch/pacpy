import numpy as np
import pacpy
import os
from pacpy.filt import firfls, morletf, firf


def test_firf():
    """
    Confirm consistency in FIR filtering
    """
    # Load data
    data = np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    assert np.allclose(
        np.sum(np.abs(firf(data, (13, 30)))), 5517466.5857, atol=10 ** -5)


def test_firfls():
    """
    Confirm consistency in FIR least-squares filtering
    """
    # Load data
    data = np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    assert np.allclose(
        np.sum(np.abs(firfls(data, (13, 30)))), 6020360.04878, atol=10 ** -5)


def test_morletf():
    """
    Confirm consistency in morlet wavelet filtering
    """
    # Load data
    data = np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    assert np.allclose(
        np.sum(np.abs(morletf(data, 21.5))), 40125678.7918, atol=10 ** -4)
