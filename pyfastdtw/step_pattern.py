# -*- coding: utf-8 -*-
import numpy as np

_MAX = np.iinfo(int).max

class BasePattern():
    """
    step pettern base class
    """
    def __init__(self):
        # number of patterns
        self.num_pattern = len(self.pattern_info)
        # max length of pattern
        self.max_pattern_len = max([len(pi["indices"]) for pi in self.pattern_info])
        self._get_array()

    def plot(self):
        #TODO
        raise NotImplementedError()

    def _get_array(self):
        """
        get pattern information as np.ndarray for numba.jit
        """
        array = np.zeros([self.num_pattern,self.max_pattern_len,3],dtype="int")
        for pidx in range(self.num_pattern):
            pattern_len = len(self.pattern_info[pidx]["indices"])
            for sidx in range(pattern_len):
                array[pidx,sidx,0:2] = self.pattern_info[pidx]["indices"][sidx]
                if sidx == 0:
                    array[pidx,sidx,2] = _MAX
                else:
                    array[pidx,sidx,2] = self.pattern_info[pidx]["weights"][sidx-1]
        self.array = array

    def normalize_cumsum_matrix(self,D):
        if not self.is_normalizable:
            return None
        if self.normalize_guide == "N+M":
            return D/sum(D.shape)
        elif self.normalize_guide == "N":
            return D/D.shape[0]
        elif self.normalize_guide == "M":
            return D/D.shape[1]
        else:
            raise Exception()

    @property
    def is_normalizable(self):
        return self.normalize_guide != "none"

class Symmetric1(BasePattern):
    pattern_info = [
        dict(
            indices=[(-1,0),(0,0)],
            weights=[1]
        ),
        dict(
            indices=[(-1,-1),(0,0)],
            weights=[1]
        ),
        dict(
            indices=[(0,-1),(0,0)],
            weights=[1]
        )
    ]
    normalize_guide = "none"

    def __init__(self):
        super().__init__()

class Symmetric2(BasePattern):
    pattern_info = [
        dict(
            indices=[(-1,0),(0,0)],
            weights=[1]
        ),
        dict(
            indices=[(-1,-1),(0,0)],
            weights=[2]
        ),
        dict(
            indices=[(0,-1),(0,0)],
            weights=[1]
        )
    ]
    normalize_guide = "N+M"

    def __init__(self):
        super().__init__()


class SymmetricP1(BasePattern):
    pattern_info = [
        dict(
            indices=[(-2,-1),(-1,0),(0,0)],
            weights=[2,1]
        ),
        dict(
            indices=[(-1,-1),(0,0)],
            weights=[2]
        ),
        dict(
            indices=[(-1,-2),(0,-1),(0,0)],
            weights=[2,1]
        )
    ]
    normalize_guide = "N+M"

    def __init__(self):
        super().__init__()



class SymmetricP2(BasePattern):
    pattern_info = [
        dict(
            indices=[(-3,-2),(-2,-1),(-1,0),(0,0)],
            weights=[2,2,1]
        ),
        dict(
            indices=[(-1,-1),(0,0)],
            weights=[2]
        ),
        dict(
            indices=[(-2,-3),(-1,-2),(0,-1),(0,0)],
            weights=[2,2,1]
        )
    ]
    normalize_guide = "N+M"

    def __init__(self):
        super().__init__()
