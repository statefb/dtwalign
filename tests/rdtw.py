# -*- coding: utf-8 -*-
"""wrapper of dtw package of R for test
"""

import gc
import numpy as np
from rpy2.robjects import r
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.rinterface import RRuntimeError

def get_robject(result, obj_name):
    for i in result.items():
        if i[0] == obj_name:
            return i[1]

class BaseDtw():
    def __init__(self,step_pattern="symmetric2",window_type=None,window_size=None,
        distance_only=False,open_end=False,open_begin=False):
        # set parameters
        self.step_pattern = step_pattern
        self.window_type = window_type
        self.window_size = window_size
        self.distance_only = distance_only
        self.open_end = open_end
        self.open_begin = open_begin

        # parameter check
        if self.window_type is not None and window_size is None:
            raise ValueError("must specify window_size if window_type is not None.")

    @staticmethod
    def get_path_matrix(num_sample,path):
        """
        input:
            num_sample: original ref or query number of samples (before alignment)
            path: ref path or query path
        """
        num_path = len(path)
        D = np.zeros([num_sample,num_path])
        for i,p in enumerate(path):
            D[p,i] = 1
        return D


class DtwR(BaseDtw):
    def __init__(self,step_pattern="symmetric2",window_type=None,window_size=10000,
        distance_only=False,open_end=False,open_begin=False,rdtw=None):
        self.step_pattern = step_pattern
        self.window_type = window_type
        self.window_size = window_size
        self.distance_only = distance_only
        self.open_end = open_end
        self.open_begin = open_begin
        # # parameter check
        # if self.window_type is not None and window_size is None:
        #     raise ValueError("must specify window_size if window_type is not None.")
        """"""
        if rdtw is None:
            # rdtw package object
            self._rdtw = importr("dtw")
        else:
            self._rdtw = rdtw
        # array conversion activation
        numpy2ri.activate()
        pandas2ri.activate()

        # set window type if it's none
        if self.window_type is None: self.window_type = "none"

    def fit(self,global_cost_matrix):
        # set params to R environment
        r.assign("window.type",self.window_type)
        r.assign("window.size",self.window_size)
        r.assign("step.pattern",DtwR._get_r_step_pattern(self._rdtw,self.step_pattern))
        r.assign("open.begin",self.open_begin)
        r.assign("open.end",self.open_end)
        r.assign("distance.only",self.distance_only)
        r.assign("lm",global_cost_matrix)
        try:
            res_dtw = r("dtw(lm,keep.internals=T,window.type=window.type,\
                distance.only=distance.only,window.size=window.size,\
                step.pattern=step.pattern,open.begin=open.begin,open.end=open.end)")
        except RRuntimeError as e:
            print("An exception occurred in rdtw package: " + e.args[0])
            raise

        gc.collect()

        self.distance = np.array(get_robject(res_dtw,"distance"))[0]
        self.normalized_distance = np.array(get_robject(res_dtw,"normalizedDistance"))[0]
        if not self.distance_only:
            self.ref_path = np.array(res_dtw[13],dtype="int") - 1
            self.ref_path_matrix = BaseDtw.get_path_matrix(
                global_cost_matrix.shape[1],self.ref_path)
            self.query_path = np.array(res_dtw[12],dtype="int") - 1
            self.query_path_matrix = BaseDtw.get_path_matrix(
                global_cost_matrix.shape[0],self.query_path)
            self.ref_warp_index = np.array(self._rdtw.warp(res_dtw,True),dtype="int") - 1
            self.query_warp_index = np.array(self._rdtw.warp(res_dtw,False),dtype="int") - 1
            self.cumsum_matrix = np.array(get_robject(res_dtw,"costMatrix"))

        return self

    @staticmethod
    def _get_r_step_pattern(rdtw,step_pattern_str):
        if step_pattern_str == "symmetric2":return rdtw.symmetric2
        elif step_pattern_str == "symmetric1":return rdtw.symmetric1
        elif step_pattern_str == "symmetricP05":return rdtw.symmetricP05
        elif step_pattern_str == "symmetricP2":return rdtw.symmetricP2
        elif step_pattern_str == "symmetricP0":return rdtw.symmetricP0
        elif step_pattern_str == "symmetricP1":return rdtw.symmetricP1
        elif step_pattern_str == "asymmetric":return rdtw.asymmetric
        elif step_pattern_str == "asymmetricP0":return rdtw.asymmetricP0
        elif step_pattern_str == "asymmetricP05":return rdtw.asymmetricP05
        elif step_pattern_str == "asymmetricP2":return rdtw.asymmetricP2
        elif step_pattern_str == "asymmetricP1":return rdtw.asymmetricP1
        elif step_pattern_str == "typeIa":return rdtw.typeIa
        elif step_pattern_str == "typeIb":return rdtw.typeIb
        elif step_pattern_str == "typeIc":return rdtw.typeIc
        elif step_pattern_str == "typeId":return rdtw.typeId
        elif step_pattern_str == "typeIas":return rdtw.typeIas
        elif step_pattern_str == "typeIbs":return rdtw.typeIbs
        elif step_pattern_str == "typeIcs":return rdtw.typeIcs
        elif step_pattern_str == "typeIds":return rdtw.typeIds
        elif step_pattern_str == "typeIIa":return rdtw.typeIIa
        elif step_pattern_str == "typeIIb":return rdtw.typeIIb
        elif step_pattern_str == "typeIIc":return rdtw.typeIIc
        elif step_pattern_str == "typeIId":return rdtw.typeIId
        elif step_pattern_str == "typeIIIc":return rdtw.typeIIIc
        elif step_pattern_str == "typeIVc":return rdtw.typeIVc
        elif step_pattern_str == "mori2006":return rdtw.mori2006
        else:raise NotImplementedError()
