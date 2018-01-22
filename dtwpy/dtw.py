# -*- coding: utf-8 -*-

from scipy.spatial.distance import cdist
from .cost import _calc_cumsum_matrix_py,_calc_cumsum_matrix_jit
from .backtrack import _backtrack_py,_backtrack_jit
from .step_pattern import *
from .window import *
from .result import DtwResult
from .distance import _get_alignment_distance

def dtw(x,y,dist="euclidean",window_type="none",window_size=None,step_pattern="symmetric2",\
    dist_only=False,open_begin=False,open_end=False,approx=False):
    """high-level dtw interface

    Parameters
    ----------
    x : 1D or 2D array (sample * feature)
        query time series
    y : 1D or 2D array (sample * feature)
        reference time series
    dist : string or function
        define how to calclulate pair-wise distance between x and y
        string : metric argument of scipy.spatial.distance
        function : user defined function. argument must be 2D array (sample * feature)
            ex) user_func(a,b) : a and b are 2D array
    window_type : string
        define window type
    window_size : int
    step_pattern : string
    dist_only : bool
    open_begin : bool
    open_end : bool
    approx : bool

    Returns
    -------
    DtwResult

    """
    len_x = x.shape[0]; len_y = y.shape[0]
    if x.ndim == 1: x = x[:,np.newaxis]
    if y.ndim == 1: y = y[:,np.newaxis]

    # get pair-wise cost matrix
    if type(dist) == str:
        X = cdist(x,y,metric=dist)
    else:
        # TODO: for efficiency, only window cell should be calculated
        X = np.zeros([len_x,len_y])
        for xidx in range(len_x):
            for yidx in range(len_y):
                X[xidx,yidx] = dist(x[xidx,:],y[yidx,:])

    return dtw_from_distance_matrix(X,window_type,window_size,step_pattern,dist_only,\
        open_begin,open_end,approx)

def dtw_from_distance_matrix(X,window_type="none",window_size=None,step_pattern="symmetric2",\
    dist_only=False,open_begin=False,open_end=False,approx=False):
    """run dtw from distance matrix

    Parameters
    ----------
    X : 2D array
        pre-computed pair-wise distance matrix
    others : see dtw function

    """
    len_x,len_y = X.shape
    window = _get_window(window_type,window_size,len_x,len_y)
    pattern = _get_pattern(step_pattern)
    return dtw_low(X,window,pattern,dist_only,open_begin,open_end,approx)

def dtw_low(X,window,pattern,dist_only=False,\
    open_begin=False,open_end=False,approx=False):
    """low-level dtw interface

    Parameters
    ----------
    X : 2D array
        pair-wise cost matrix
    window : dtwpy.window.BaseWindow object
        window object
    pattern : dtwpy.step_pattern.BasePattern object
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
        raise ValueError("pair-wise cost matrix must NOT contain negative values")
    if not isinstance(window,BaseWindow):
        raise ValueError("window argument must be Window object")
    if not isinstance(pattern,BasePattern):
        raise ValueError("pattern argument must be Pattern object")
    if open_begin:
        if not pattern.normalize_guide == "N":
            raise ValueError("open-begin alignment requires 'N' normalizable step pattern")
    if open_end:
        if not pattern.is_normalizable:
            raise ValueError("open-end alignment requires normalizable step pattern")

    D = _calc_cumsum_matrix_jit(X,window.list,pattern.array,open_begin)
    dist,normalized_dist,last_idx = _get_alignment_distance(D,pattern,open_begin,open_end)

    if dist_only:
        path = None
        if open_begin:
            D = D[1:,:]
    else:
        path = _backtrack_jit(D,pattern.array,last_idx)
        if open_begin:
            D = D[1:,:]
            path = path[1:,:]
            path[:,0] -= 1

    result = DtwResult(D,path,window,pattern)
    # set some properties
    result.distance = dist
    result.normalized_distance = normalized_dist

    return result

def _get_window(window_type,window_size,len_x,len_y):
    if window_type == "sakoechiba":
        return SakoechibaWindow(len_x,len_y,window_size)
    elif window_type == "itakura":
        return ItakuraWindow(len_x,len_y)
    elif window_type == "none":
        return NoWindow(len_x,len_y)
    else:
        raise NotImplementedError("given window type not supported")

def _get_pattern(pattern_str):
    if pattern_str == "symmetric1":
        return Symmetric1()
    elif pattern_str == "symmetric2":
        return Symmetric2()
    elif pattern_str == "symmetricP05":
        return SymmetricP05()
    elif pattern_str == "symmetricP0":
        return SymmetricP0()
    elif pattern_str == "symmetricP1":
        return SymmetricP1()
    elif pattern_str == "symmetricP2":
        return SymmetricP2()
    elif pattern_str == "asymmetric":
        return Asymmetric()
    elif pattern_str == "asymmetricP0":
        return AsymmetricP0()
    elif pattern_str == "asymmetricP05":
        return AsymmetricP05()
    elif pattern_str == "asymmetricP1":
        return AsymmetricP1()
    elif pattern_str == "asymmetricP2":
        return AsymmetricP2()
    else:
        raise NotImplementedError("given step pattern not supported")
