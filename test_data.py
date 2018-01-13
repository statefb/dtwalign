# -*- coding: utf-8 -*-
"""test data generator
"""

import numpy as np

np.random.seed(1234)

def gen_data(open_bigen,open_end):
    if (not open_bigen) and (not open_end):
        y1 = np.sin(2*np.pi*3*np.linspace(0,1,120))
        y1 += np.random.rand(y1.size)
        x1 = np.sin(2*np.pi*3.1*np.linspace(0,1,101))
        x1 += np.random.rand(x1.size)
        # x:reference, y:query
        X = np.abs(x1[:,np.newaxis] - y1[np.newaxis,:])
        return x1,y1,X
    elif (not open_bigen) and (open_end):
        y1 = np.sin(2*np.pi*3*np.linspace(0,2,240))
        y1 += np.random.rand(y1.size)
        x1 = np.sin(2*np.pi*3.1*np.linspace(0,1,101))
        x1 += np.random.rand(x1.size)
        # x:reference, y:query
        X = np.abs(x1[:,np.newaxis] - y1[np.newaxis,:])
        return x1,y1,X
    elif (open_bigen) and (not open_end):
        raise NotImplementedError()
    else:
        raise NotImplementedError()
