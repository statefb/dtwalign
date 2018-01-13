# -*- coding: utf-8 -*-
import numpy as np

def _get_alignment_distance(D,pattern,open_end):
    dist = D[-1,-1]
    normalized_dist = None
    last_idx = -1
    
    if pattern.is_normalizable:
        # get the last row of D
        last_row = D[-1,:]
        normalized_last_row = pattern._normalize(\
            last_row,D.shape[0],D.shape[1])
        if open_end:
            # if open-end, find index of last row
            # that minimize alignment cost
            last_idx = np.argmin(normalized_last_row)
            dist = last_row[last_idx]
            normalized_dist = normalized_last_row[last_idx]
        else:
            normalized_dist = last_row[-1]

    # check whether path can reach at end point with given constraint
    if dist == np.inf:
        raise ValueError("no alignment path found")
    return dist,normalized_dist,last_idx
