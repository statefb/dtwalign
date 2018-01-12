import numpy as np
from numpy.testing import assert_almost_equal
from pyfastdtw import dtw,dtw_low
from pyfastdtw.step_pattern import *
from pyfastdtw.window import *
from pyfastdtw.cost import _calc_cumsum_matrix_jit
from pyfastdtw.backtrack import _backtrack_py,_backtrack_jit

np.random.seed(1234)

x1 = np.sin(2*np.pi*3*np.linspace(0,1,120))
x1 += np.random.rand(x1.size)
y1 = np.sin(2*np.pi*3.1*np.linspace(0,1,101))
y1 += np.random.rand(y1.size)

plt.figure()
plt.plot(x1)
plt.plot(y1)

# x:reference, y:query
X = np.abs(x1[:,np.newaxis] - y1[np.newaxis,:])
# window = SakoechibaWindow(X.shape[0],X.shape[1],size=20)
window = NoWindow(X.shape[0],X.shape[1])

pattern = AsymmetricP2()
# res = dtw_low(X,window,pattern)

D = _calc_cumsum_matrix_jit(X,window.list,pattern.array)

path_py = _backtrack_py(D,pattern)
path_jit = _backtrack_jit(D,pattern.array)

# print(path_py)
# print(path_jit)

print(path_py.shape)
print(path_jit.shape)

# assert if equal to naive
assert_almost_equal(path_py,path_jit)

plt.figure()
plt.plot(path_py,path_jit)
plt.show()
