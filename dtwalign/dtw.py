# -*- coding: utf-8 -*-
"""A complehensive dynamic time warping package.

The MIT License (MIT)

Copyright (c) 2018 statefb.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np
from scipy.spatial.distance import cdist
from .cost import _calc_cumsum_matrix_jit
from .backtrack import _backtrack_jit
from .step_pattern import *
from .window import *
from .result import DtwResult
from .distance import _get_alignment_distance


def dtw(x, y, dist="euclidean", window_type="none", window_size=None,
    step_pattern="symmetric2", dist_only=False, open_begin=False, open_end=False):
    """Perform dtw.

    Parameters
    ----------
    x : 1D or 2D array (sample * feature)
        Query time series.

    y : 1D or 2D array (sample * feature)
        Reference time series.

    dist : string or function
        Define how to calclulate pair-wise distance between x and y.
        string - metric argument of scipy.spatial.distance
        function - user function that defines metric between two samples.
            ex) euclidean distance: user_func = lambda x,y : np.sqrt((x-y)**2)

    window_type : string
        Window type to use.
        "sakoechiba" - Sakoechiba window
        "itakura" - Itakura window

    window_size : int
        Window size to use for Sakoechiba window.

    step_pattern : string
        Step pattern to use.

    dist_only : bool
        Whether or not to obtain warping path. If true,
        only alignment distance will be calculated.

    open_begin : bool
        Whether or not perform open-ended alignment at the starting point of
        query time series. If true, partial alignment will be performed.

    open_end : bool
        Whether or not perform open-ended alignment at the end point of
        query time series. If true, partial alignment will be performed.

    """
    len_x = x.shape[0]; len_y = y.shape[0]
    # if 1D array, convert to 2D array
    if x.ndim == 1: x = x[:, np.newaxis]
    if y.ndim == 1: y = y[:, np.newaxis]

    # get pair-wise cost matrix
    if type(dist) == str:
        # scipy
        X = cdist(x, y, metric=dist)
    else:
        # user defined metric
        window = _get_window(window_type, window_size, len_x, len_y)
        X = np.ones([len_x, len_y]) * np.inf
        for i, j in window.list:
            X[i, j] = dist(x[i, :], y[j, :])

    return dtw_from_distance_matrix(X, window_type, window_size, step_pattern,
        dist_only, open_begin, open_end)


def dtw_from_distance_matrix(X, window_type="none", window_size=None,
    step_pattern="symmetric2", dist_only=False, open_begin=False, open_end=False):
    """Perform dtw using pre-computed pair-wise distance matrix.

    Parameters
    ----------
    X : 2D array
        pre-computed pair-wise distance matrix

    others : see dtw function

    """
    len_x, len_y = X.shape
    window = _get_window(window_type, window_size, len_x, len_y)
    pattern = _get_pattern(step_pattern)
    return dtw_low(X, window, pattern, dist_only, open_begin, open_end)


def dtw_low(X, window, pattern, dist_only=False,
    open_begin=False, open_end=False):
    """Low-level dtw interface.

    Parameters
    ----------
    X : 2D array
        pair-wise distance matrix

    window : dtwalign.window.BaseWindow object
        window object

    pattern : dtwalign.step_pattern.BasePattern object
        step pattern object

    others : see dtw function

    """
    # validation
    if X[X < 0].sum() != 0:
        raise ValueError("pair-wise cost matrix must NOT contain negative values")
    if not isinstance(window, BaseWindow):
        raise ValueError("window argument must be Window object")
    if not isinstance(pattern, BasePattern):
        raise ValueError("pattern argument must be Pattern object")
    if open_begin:
        if not pattern.normalize_guide == "N":
            raise ValueError("open-begin alignment requires 'N' normalizable step pattern")
    if open_end:
        if not pattern.is_normalizable:
            raise ValueError("open-end alignment requires normalizable step pattern")

    # compute cumsum distance matrix
    D = _calc_cumsum_matrix_jit(X, window.list, pattern.array, open_begin)
    # get alignment distance
    dist, normalized_dist, last_idx = _get_alignment_distance(D, pattern,
        open_begin, open_end)

    if dist_only:
        path = None
        if open_begin:
            D = D[1:, :]
    else:
        # backtrack to obtain warping path
        path = _backtrack_jit(D, pattern.array, last_idx)
        if open_begin:
            D = D[1:, :]
            path = path[1:, :]
            path[:, 0] -= 1

    result = DtwResult(D, path, window, pattern)
    # set distance properties
    result.distance = dist
    result.normalized_distance = normalized_dist

    return result


def _get_window(window_type, window_size, len_x, len_y):
    if window_type == "sakoechiba":
        return SakoechibaWindow(len_x, len_y, window_size)
    elif window_type == "itakura":
        return ItakuraWindow(len_x, len_y)
    elif window_type == "none":
        return NoWindow(len_x, len_y)
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
    elif pattern_str == "typeIa":
        return TypeIa()
    elif pattern_str == "typeIb":
        return TypeIb()
    elif pattern_str == "typeIc":
        return TypeIc()
    elif pattern_str == "typeId":
        return TypeId()
    elif pattern_str == "typeIas":
        return TypeIas()
    elif pattern_str == "typeIbs":
        return TypeIbs()
    elif pattern_str == "typeIcs":
        return TypeIcs()
    elif pattern_str == "typeIds":
        return TypeIds()
    elif pattern_str == "typeIIa":
        return TypeIIa()
    elif pattern_str == "typeIIb":
        return TypeIIb()
    elif pattern_str == "typeIIc":
        return TypeIIc()
    elif pattern_str == "typeIId":
        return TypeIId()
    elif pattern_str == "typeIIIc":
        return TypeIIIc()
    elif pattern_str == "typeIVc":
        return TypeIVc()
    elif pattern_str == "mori2006":
        return Mori2006()
    elif pattern_str == "unitary":
        return Unitary()
    else:
        raise NotImplementedError("given step pattern not supported")
