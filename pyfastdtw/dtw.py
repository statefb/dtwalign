# -*- coding: utf-8 -*-

from .cost import _calc_cumsum_matrix_py,_calc_cumsum_matrix_jit
from .backtrack import _backtrack_py,_backtrack_jit
from .step_pattern import *
from .window import *
from .result import DtwResult

def dtw(x,y,dist,window_type,window_size,step_pattern="symmetric2",\
    dist_only=True,fast=True,open_begin=False,open_end=False):
    """high-level dtw interface
    TODO:
    ・x,y,distの代わりにXを受け取るようにできる
    ・window関数，マトリクスを受け取るようにできる
    """
    # get pair-wise cost matrix
    # X = dist(x,y)

    widow = _get_window(window_type,window_size,x.shape[0],y.shape[0])

    # get step pattern
    # pattern = get_pattern(step_pattern)
    return dtw_low(X,window,pattern,dist_only,fast,open_begin,open_end)

def dtw_low(X,window,pattern,dist_only=False,fast=True,\
    open_begin=False,open_end=False):
    """low-level dtw interface

    Parameters
    ----------
    X : pair-wise cost matrix
    window : windowing function
    pattern : step pattern
    dist_only : if true, only alignment cost will be calculated
    fast : if true, use fast-dtw
    open_begin :
    open_end :

    Returns
    -------
    result : DtwResult

    Notes
    -----

    """
    # validation
    if X[X < 0].sum() != 0:
        raise ValueError("pair-wise cost matrix must NOT have negative value")

    # naive implementation
    # D = _calc_cumsum_matrix_py(X,window,pattern)
    # fast implementation
    D = _calc_cumsum_matrix_jit(X,window.list,pattern.array)

    if dist_only:
        path = None
    else:
        # naive
        # path = _backtrack_py(D,pattern)
        # fast
        path = _backtrack_jit(D,pattern.array)

    result = DtwResult(D,path,window,pattern)

    return result

def _get_window(window_type,window_size,len_x,len_y):
    if window_type == "sakoechiba":
        window = SakoechibaWindow(len_x,len_y,window_size)
    else:
        window = NoWindow(len_x,len_y)
    return window
