# pacpy
[![Build Status](https://travis-ci.org/voytekresearch/pacpy.svg)](https://travis-ci.org/voytekresearch/pacpy)

A module to calculate phase-amplitude coupling in Python.

## Demo

A [Binder](http://mybinder.org) demo, complete with simulated data, can be found [here](https://github.com/srcole/pacpybinder).

## Install

	pip install pacpy

Tested on Linux (Ubuntu 4.10), OS X (10.10.4), and Windows 9.

## Dependencies

- numpy
- scipy
- pytest (optional)

That is , we assume [Anaconda](https://store.continuum.io/cshop/anaconda/) is installed. 

## Matlab

The wrapper for MATLAB can be found at, https://github.com/voytekresearch/pacmat

## Usage

An example of calculating PAC from two simulated voltage signals using the phase-locking value (PLV) method:

        >>> import numpy as np
		>>> from scipy.signal import hilbert
		>>> from pacpy.pac import plv
		>>> t = np.arange(0, 10, .001) # Define time array
		>>> lo = np.sin(t * 2 * np.pi * 6) # Create low frequency carrier
		>>> hi = np.sin(t * 2 * np.pi * 100) # Create modulated oscillation
		>>> hi[np.angle(hilbert(lo)) > -np.pi*.5] = 0 # Clip to 1/4 of cycle
		>>> plv(lo, hi, (4,8), (80,150)) # Calculate PAC
		0.99863308613553081
