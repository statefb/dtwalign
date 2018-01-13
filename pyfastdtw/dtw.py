# -*- coding: utf-8 -*-

from .cost import _calc_cumsum_matrix_py,_calc_cumsum_matrix_jit
from .backtrack import _backtrack_py,_backtrack_jit
from .step_pattern import *
from .window import *
from .result import DtwResult
from .distance import _get_alignment_distance

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
    return dtw_low(X,window,pattern,dist_only,approx,open_begin,open_end)

def dtw_low(X,window,pattern,dist_only=False,approx=True,\
    open_begin=False,open_end=False):
    """low-level dtw interface

    Parameters
    ----------
    X : 2D array
        pair-wise cost matrix
    window : pyfastdtw.window.BaseWindow object
        window object
    pattern : pyfastdtw.step_pattern.BasePattern object
        step pattern object
    dist_only : bool
        if true, only alignment cost will be calculated
    approx : bool
        if true, use fast-dtw
    open_begin : bool
    open_end : bool

    Returns
    -------
    result : DtwResult

    Notes
    -----

    """
    # validation
    if X[X < 0].sum() != 0:
        raise ValueError("pair-wise cost matrix must NOT have negative value")
    if not isinstance(window,BaseWindow):
        raise ValueError("window argument must be Window object")
    if not isinstance(pattern,BasePattern):
        raise ValueError("pattern argument must be Pattern object")
    if open_begin:
        if not pattern.normalize_guide == "N":
            raise ValueError("open-begin alignment requires step pattern \
                that has 'N' normalization. see original paper for detail.")
    if open_end:
        if not pattern.is_normalizable:
            raise ValueError("open-end alignment requires normalizable step pattern")

    # naive implementation
    # D = _calc_cumsum_matrix_py(X,window,pattern)
    # fast implementation
    D = _calc_cumsum_matrix_jit(X,window.list,pattern.array)
    dist,normalized_dist,last_idx = _get_alignment_distance(D,pattern,open_end)

    if dist_only:
        path = None
    else:
        # naive
        # path = _backtrack_py(D,pattern)
        # approx
        path = _backtrack_jit(D,pattern.array,last_idx)

    result = DtwResult(D,path,window,pattern)
    # set some properties
    result.distance = dist
    result.normalized_distance = normalized_dist

    return result

def _get_window(window_type,window_size,len_x,len_y):
    if window_type == "sakoechiba":
        window = SakoechibaWindow(len_x,len_y,window_size)
    else:
        window = NoWindow(len_x,len_y)
    return window
