import numpy as np
from pyfastdtw import dtw,dtw_low
from pyfastdtw.step_pattern import *
from pyfastdtw.window import *

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

patterns = [
    Symmetric1(),
    Symmetric2(),
    SymmetricP1(),
    SymmetricP2(),
    SymmetricP05(),
    SymmetricP0(),
    Asymmetric(),
    AsymmetricP1(),
    AsymmetricP2(),
    AsymmetricP0(),
    AsymmetricP05()
]

for pattern in patterns:
    res = dtw_low(X,window,pattern)
    res.plot_path("cum")

    # get path
    xp = res.path[:,0]
    yp = res.path[:,1]
    # warped data
    x1w = x1[xp]
    y1w = y1[yp]

    # plot warped data(both)
    plt.figure()
    plt.plot(x1w)
    plt.plot(y1w)
    plt.show()

# plot path separately
# plt.figure()
# plt.plot(xp);plt.plot(yp)

"""R"""
# from mtsa.common.dtw import DtwR
# dtwr = DtwR(step_pattern="asymmetricP2")
# dtwr.fit(X)
