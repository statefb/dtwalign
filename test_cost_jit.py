import numpy as np
from numpy.testing import assert_almost_equal
from pyfastdtw.cost import _calc_cumsum_matrix_py
from pyfastdtw.cost import _calc_cumsum_matrix_jit
from pyfastdtw.window import *
from pyfastdtw.step_pattern import *

x1 = np.sin(2*np.pi*3*np.linspace(0,1,1000))
x1 += np.random.rand(x1.size)
y1 = np.sin(2*np.pi*3.1*np.linspace(0,1,1001))
y1 += np.random.rand(y1.size)
X = x1[:,np.newaxis] - y1[np.newaxis,:]

sp = SymmetricP1()
win = NoWindow(x1.size,y1.size)

Djit = _calc_cumsum_matrix_jit(X,win.list,sp.array)
Dpy = _calc_cumsum_matrix_py(X,win,sp)

assert_almost_equal(Dpy,Djit)
