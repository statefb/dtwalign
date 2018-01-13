# -*- coding: utf-8 -*-
"""jit関数からjit関数を呼び出すテスト
"""
from numba import jit

@jit(nopython=True)
def f1(x):
    res = 0
    for i in range(10000):
        res += f2(x)
    return res

@jit(nopython=True)
def f2(x):
    res = 0
    for i in range(1000):
        res += i + x
    return res



def f1_py(x):
    res = 0
    for i in range(10000):
        res += f2(x)
    return res

def f2_py(x):
    res = 0
    for i in range(1000):
        res += i + x
    return res
