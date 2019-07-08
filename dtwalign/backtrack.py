# -*- coding: utf-8 -*-
"""Backtracking implementation."""

import numpy as np
from numba import jit


@jit(nopython=True)
def _backtrack_jit(D, p_ar, last_idx=-1):
    """Fast implementation by numba.jit.

    D : 2D array
        cumsum cost matrix
    p_ar : 3D array
        step pattern array (see step_pattern.py)
    """
    # number of patterns
    num_pattern = p_ar.shape[0]
    # initialize index
    i, j = D.shape
    i -= 1
    if last_idx == -1:
        j -= 1
    else:
        j = last_idx
    # alignment path
    path = np.array(((i, j),), dtype=np.int64)
    # cache to memorize D
    D_cache = np.ones(num_pattern, dtype=np.float64) * np.inf

    while True:
        if i == 0 and j == 0:
            break
        for pidx in range(num_pattern):
            # get D value corresponds to end of pattern node
            pattern_index = p_ar[pidx, 0, 0:2]
            ii = int(i + pattern_index[0])
            jj = int(j + pattern_index[1])
            if ii < 0 or jj < 0:
                D_cache[pidx] = np.inf
            else:
                D_cache[pidx] = D[ii, jj]

        if (D_cache == np.inf).all():
            # break if there is no direction can be taken
            break

        # find path minimize D_chache
        min_pattern_idx = np.argmin(D_cache)
        # get where pattern passed
        path_to_add = _get_local_path(D, p_ar[min_pattern_idx, :, :], i, j)
        # concatenate
        path = np.vstack((path, path_to_add))

        i += p_ar[min_pattern_idx, 0, 0]
        j += p_ar[min_pattern_idx, 0, 1]

    return path[::-1]


@jit(nopython=True)
def _get_local_path(D, p_ar, i, j):
    """Helper function to get local path.
    D : cumsum matrix
    p_ar : array of pattern that minimize D at i,j
    """
    weight_col = p_ar[:, 2]
    step_selector = np.where(weight_col != 0)[0]
    # note: starting point of pattern was already added
    step_selector = step_selector[:-1]
    # initialize local path
    local_path = np.ones((step_selector.size, 2),
        dtype=np.int64) * -1
    for sidx in step_selector:
        # memorize where passed
        pattern_index = p_ar[sidx, 0:2]

        ii = int(i + pattern_index[0])
        jj = int(j + pattern_index[1])

        local_path[sidx, :] = (ii, jj)
    return local_path[::-1]
