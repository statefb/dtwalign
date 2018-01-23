# -*- coding: utf-8 -*-
"""test data generator
"""

import numpy as np
from scipy.spatial.distance import cdist

np.random.seed(1234)

def gen_data(open_begin,open_end):
    if (not open_begin) and (not open_end):
        y1 = np.sin(2*np.pi*3*np.linspace(0,1,120))
        y1 += np.random.rand(y1.size)
        x1 = np.sin(2*np.pi*3.1*np.linspace(0,1,101))
        x1 += np.random.rand(x1.size)
        X = cdist(x1[:,np.newaxis],y1[:,np.newaxis],metric="euclidean")
        # X = np.abs(x1[:,np.newaxis] - y1[np.newaxis,:])
        return x1,y1,X
    elif (not open_begin) and (open_end):
        y1 = np.sin(2*np.pi*3*np.linspace(0,2,240))
        y1 += np.random.rand(y1.size)
        x1 = np.sin(2*np.pi*3.1*np.linspace(0,1,101))
        x1 += np.random.rand(x1.size)
        X = cdist(x1[:,np.newaxis],y1[:,np.newaxis],metric="euclidean")
        return x1,y1,X
    elif (open_begin) and (not open_end):
        y1 = np.sin(2*np.pi*3*np.linspace(0,2,240))
        y1 += np.random.rand(y1.size)
        x1 = np.sin(2*np.pi*3.1*np.linspace(0,1,101))
        x1 += np.random.rand(x1.size)
        X = cdist(x1[:,np.newaxis],y1[:,np.newaxis],metric="euclidean")
        return x1,y1,X
    else:
        y1 = np.sin(2*np.pi*2*np.linspace(0,1,120))
        y1 += np.random.rand(y1.size)
        x1 = np.sin(2*np.pi*2.1*np.linspace(0.3,0.8,100))
        x1 += np.random.rand(x1.size)
        X = cdist(x1[:,np.newaxis],y1[:,np.newaxis],metric="euclidean")
        return x1,y1,X

def gen_csv(open_begin,open_end):
    x,y,X = gen_data(open_begin,open_end)
    np.savetxt("ref.csv",y,delimiter=",")
    np.savetxt("query.csv",x,delimiter=",")
    np.savetxt("X.csv",X,delimiter=",")
