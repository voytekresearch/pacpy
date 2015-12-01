import pytest
import numpy as np
import pacpy
import os
from pacpy.pac import plv, glm, mi_tort, mi_canolty, ozkurt, otc, _peaktimes, _chunk_time, comodulogram, pa_series, pa_dist
from pacpy.filt import butterf, firf
from scipy.signal import hilbert
import copy


def genPAC1(phabias=.5, flo=5, fhi=100, glm_bias=False):
    """
    Generate two signals that have very high PAC
    """
    dt = .001
    T = 10
    t = np.arange(0, T, dt)

    lo = np.sin(t * 2 * np.pi * flo)
    hi = np.sin(t * 2 * np.pi * fhi)

    if glm_bias:
        hi = hi * (lo + 1)
    else:
        pha = np.angle(hilbert(lo))
        hi[pha > -np.pi + phabias] = 0

    return lo, hi


def genPAC0(flo=5, fhi=100):
    """
    Generate two signals that have very low PAC
    """
    dt = .001
    T = 10
    t = np.arange(0, T, dt)

    lo = np.sin(t * 2 * np.pi * flo)
    hi = np.sin(t * 2 * np.pi * fhi)
    return lo, hi


def ident(x, f, fs):
    return x


def test_plv():
    """
    Test PAC function: PLV.
    1. Confirm consistency of output with example data
    2. Confirm consistency of output with example data using iir filter
    3. Confirm PAC=1 when expected
    4. Confirm PAC=0 when expected
    """
    # Load data
    data = np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    assert np.allclose(
        plv(data, data, (13, 30), (80, 200)), 0.23778, atol=10 ** -5)
    assert np.allclose(
        plv(data, data, (13, 30), (80, 200), filterfn=butterf), 0.24696, atol=10 ** -5)

    # Test that the PLV function outputs close to 0 and 1 when expected
    lo, hi = genPAC1()
    assert plv(lo, hi, (4, 6), (90, 110)) > 0.99

    lo, hi = genPAC0()
    assert plv(lo, hi, (4, 6), (90, 110)) < 0.05
    
    # Test that Filterfn = False works as expected
    datalo = firf(data, (13,30))
    datahi = firf(data, (80,200))
    datahiamp = np.abs(hilbert(datahi))
    datahiamplo = firf(datahiamp, (13,30))
    pha1 = np.angle(hilbert(datalo))
    pha2 = np.angle(hilbert(datahiamplo))
    assert np.allclose(
        plv(pha1, pha2, (13, 30), (80, 200), filterfn=False),
        plv(data, data, (13, 30), (80, 200)), atol=10 ** -5)
    

def test_glm():
    """
    Test PAC function: GLM
    1. Confirm consistency of output with example data
    2. Confirm consistency of output with example data using iir filter
    3. Confirm PAC=1 when expected
    4. Confirm PAC=0 when expected
    """
    # Load data
    data = np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    assert np.allclose(
        glm(data, data, (13, 30), (80, 200)), 0.03191, atol=10 ** -5)
    assert np.allclose(
        glm(data, data, (13, 30), (80, 200), filterfn=butterf), 0.03476, atol=10 ** -5)

    # Test that the GLM function outputs close to 0 and 1 when expected
    lo, hi = genPAC1(glm_bias=True)
    assert glm(lo, hi, (4, 6), (90, 110)) > 0.99

    lo, hi = genPAC0()
    assert glm(lo, hi, (4, 6), (90, 110)) < 0.01
    
    # Test that Filterfn = False works as expected
    datalo = firf(data, (13,30))
    datahi = firf(data, (80,200))
    pha = np.angle(hilbert(datalo))
    amp = np.abs(hilbert(datahi))
    assert np.allclose(
        glm(pha, amp, (13, 30), (80, 200), filterfn=False),
        glm(data, data, (13, 30), (80, 200)), atol=10 ** -5)


def test_mi_tort():
    """
    Test PAC function: Tort MI
    1. Confirm consistency of output with example data
    2. Confirm consistency of output with example data using iir filter
    3. Confirm PAC=1 when expected
    4. Confirm PAC=0 when expected
    """
    # Load data
    data = np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    assert np.allclose(
        mi_tort(data, data, (13, 30), (80, 200)), 0.00366, atol=10 ** -5)
    assert np.allclose(mi_tort(
        data, data, (13, 30), (80, 200), filterfn=butterf), 0.00429, atol=10 ** -5)

    # Test that the Tort MI function outputs close to 0 and 1 when expected
    lo, hi = genPAC1(phabias=.2, fhi=300)
    assert mi_tort(lo, hi, (4, 6), (100, 400)) > 0.8

    lo, hi = genPAC0()
    assert mi_tort(lo, hi, (4, 6), (90, 110)) < 10 ** -5
    
    # Test that Filterfn = False works as expected
    datalo = firf(data, (13,30))
    datahi = firf(data, (80,200))
    pha = np.angle(hilbert(datalo))
    amp = np.abs(hilbert(datahi))
    assert np.allclose(
        mi_tort(pha, amp, (13, 30), (80, 200), filterfn=False),
        mi_tort(data, data, (13, 30), (80, 200)), atol=10 ** -5)


def test_mi_canolty():
    """
    Test PAC function: Canolty MI
    1. Confirm consistency of output with example data
    """
    # Load data
    data = np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    assert np.allclose(
        mi_canolty(data, data, (13, 30), (80, 200)), 19.75624, atol=10 ** -5)

    
    # Test that Filterfn = False works as expected
    datalo = firf(data, (13,30))
    datahi = firf(data, (80,200))
    pha = np.angle(hilbert(datalo))
    amp = np.abs(hilbert(datahi))
    assert np.allclose(
        mi_canolty(pha, amp, (13, 30), (80, 200), filterfn=False),
        mi_canolty(data, data, (13, 30), (80, 200)), atol=10 ** -5)


def test_ozkurt():
    """
    Test PAC function: Ozkurt
    1. Confirm consistency of output with example data
    2. Confirm consistency of output with example data using iir filter
    3. Confirm PAC=1 when expected
    4. Confirm PAC=0 when expected
    """
    # Load data
    data = np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    assert np.allclose(
        ozkurt(data, data, (13, 30), (80, 200)), 0.07548, atol=10 ** -5)
    assert np.allclose(
        ozkurt(data, data, (13, 30), (80, 200), filterfn=butterf), 0.07555, atol=10 ** -5)

    # Test that the Ozkurt PAC function outputs close to 0 and 1 when expected
    lo, hi = genPAC1(phabias=.2, fhi=300)
    hif = firf(hi, (100, 400))
    amp = np.abs(hilbert(hif))
    weight = (np.sqrt(len(amp)) * np.sqrt(np.sum(amp ** 2))) / np.sum(amp)
    assert ozkurt(lo, hi, (4, 6), (100, 400)) * weight > 0.99

    lo, hi = genPAC0()
    assert ozkurt(lo, hi, (4, 6), (90, 110)) < 0.001
    
    # Test that Filterfn = False works as expected
    datalo = firf(data, (13,30))
    datahi = firf(data, (80,200))
    pha = np.angle(hilbert(datalo))
    amp = np.abs(hilbert(datahi))
    assert np.allclose(
        ozkurt(pha, amp, (13, 30), (80, 200), filterfn=False),
        ozkurt(data, data, (13, 30), (80, 200)), atol=10 ** -5)


def test_otc():
    """
    Test PAC function: OTC
    """
    # Load data
    data = np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')

    # Confirm consistency in result
    t_modsig = (-.5, .5)
    fs = 1000.
    f_hi = (80, 200)
    f_step = 4
    pac, tf, a_events, mod_sig = otc(
        data, f_hi, f_step, fs=fs, t_modsig=t_modsig)
    assert np.allclose(pac, 220.32563, atol=10 ** -5)

    # Confirm correct shapes in outputs
    assert np.shape(tf) == ((f_hi[1] - f_hi[0]) / f_step, len(data))
    for i in range(len(a_events)):
        assert np.max(a_events[i]) <= len(data)
    assert np.shape(mod_sig) == (
        (f_hi[1] - f_hi[0]) / f_step, len(np.arange(t_modsig[0], t_modsig[1], 1 / fs)))


def test_peaktimes():
    """
    Test OTC helper function: _peaktimes
    """
    # Load data
    data = np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')

    # Confirm functionality
    assert _peaktimes(data[:1000], prc=99) == 344
    assert len(_peaktimes(data[:10000], prc=99)) == 11


def test_chunktime():
    """
    Test OTC helper function: _chunk_time
    """
    assert np.array_equal(_chunk_time(
        [5, 6, 7, 8, 10, 55, 56], samp_buffer=0), np.array([[5,  8], [10, 10], [55, 56]]))
    assert np.array_equal(_chunk_time(
        [5, 6, 7, 8, 10, 55, 56], samp_buffer=2), np.array([[5,  10], [55, 56]]))


def test_comod():
    """
    Test comodulogram function
    """
    # Load data
    data = np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    p_range = [10, 21]
    a_range = [50, 150]
    dp = 5
    da = 50
    a = comodulogram(data, data, p_range, a_range, dp, da)
    assert np.allclose(a[0][0], 0.00287, atol=10 ** -5)
    assert np.shape(a) == (len(np.arange(p_range[0], p_range[1], dp)), len(
        np.arange(a_range[0], a_range[1], da)))


def test_paseries():
    """
    Test calculation of phase and amplitude time series
    """
    # Load data
    data = np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')

    # Confirm returns correct size
    p, a = pa_series(data, data, (13, 30), (80, 200))
    assert np.shape(p) == np.shape(data)
    assert np.shape(a) == np.shape(data)

    # Confirm consistency
    assert np.allclose(np.mean(a), 11.43970, atol=10 ** -5)
    assert np.allclose(p[0], 1.57119, atol=10 ** -5)


def test_padist():
    """
    Test calculation of amplitude distribution as a function of phase
    """
    # Load data
    data = np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')

    np.random.seed(0)
    Nbins = np.random.randint(2, 20)
    pha, amp = pa_series(data, data, (13, 30), (80, 200))
    boundaries, dist = pa_dist(pha, amp, Nbins=Nbins)
    assert len(dist) == Nbins
    assert len(boundaries) == Nbins

    # Confirm consistency
    _, dist = pa_dist(pha, amp, Nbins=10)
    assert np.allclose(dist[0], 12.13961, atol=10 ** -5)


def test_raiseinputerrors():
    """
    Confirm that ValueErrors from dumb user input are raised
    """
    # Load data
    data = np.load(os.path.dirname(pacpy.__file__) + '/tests/exampledata.npy')
    data2 = copy.copy(data)
    data2[-1] = np.nan

    with pytest.raises(ValueError) as excinfo:
        plv(data, data[:-1], (13, 30), (80, 200))
    assert 'same length' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        plv(data, data2, (13, 30), (80, 200))
    assert 'NaNs' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        plv(data, data, (13, 30, 31), (80, 200))
    assert 'two elements' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        plv(data, data, (13, 30), (80, 200, 201))
    assert 'two elements' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        plv(data, data, (-13, 30), (80, 200))
    assert 'must be > 0' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        plv(data, data, (13, 30), (-80, 200))
    assert 'must be > 0' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        mi_tort(data, data, (13, 30), (80, 200), Nbins=1)
    assert 'integer >1' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        mi_tort(data, data, (13, 30), (80, 200), Nbins=8.8)
    assert 'integer >1' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        otc(data, (80, 200), -1)
    assert 'positive number' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        otc(data, (80, 200), 4, t_modsig=(.5, -.5))
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
