import numpy as np
from pacpy.pac import otc

'''
Questions/todo:

'''

def test_otc1():
    data=np.load('C:/gh/bv/pacpy/tests/exampledata.npy')
    pac, _, _, _ = otc(data, (80,200), 4, fs=1000)
    assert pac == 222.57032020274431
    
    
# Confirm my outputs are right size?
    #otc helper functions
    #comodulogram
    #phase-amplitude time series
    #phase-amplitude distribution (generate random integer to assure lengths are right)
# Confirmt hat my exceptions are actually raised
# Make sure our function outputs are right size, consistent with inputs?