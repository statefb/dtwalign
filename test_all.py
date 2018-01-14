import numpy as np
from pyfastdtw import dtw,dtw_low
from pyfastdtw.step_pattern import *
from pyfastdtw.window import *
from test_data import gen_data

#data_set
open_begin_d = True
open_end_d = False

# dtw param
open_begin = True
open_end = False

x1,y1,X = gen_data(open_begin=open_begin_d,open_end=open_end_d)

# window = SakoechibaWindow(X.shape[0],X.shape[1],size=50)
window = NoWindow(X.shape[0],X.shape[1])

plt.figure()
plt.plot(x1)
plt.plot(y1)
plt.show()

patterns = [
    # Symmetric1(),
    # Symmetric2(),
    # SymmetricP1(),
    # SymmetricP2(),
    # SymmetricP05(),
    # SymmetricP0(),
    Asymmetric(),
    # AsymmetricP1(),
    # AsymmetricP2(),
    # AsymmetricP0(),
    # AsymmetricP05()
]

for pattern in patterns:
    res = dtw_low(X,window,pattern,open_begin=open_begin,\
        open_end=open_end)
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
