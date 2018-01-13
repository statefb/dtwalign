# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

class DtwResult():
    def __init__(self,cumsum_matrix,path,window,pattern):
        self.cumsum_matrix = cumsum_matrix

        if path is None:
            self.dist_only = True
        else:
            self.dist_only = False
            self.path = path

        self._window = window
        self._pattern = pattern

    def get_warping_path(self,target="query"):
        """get warping path

        Parameters
        ----------
        target : "query" or "reference"
            specify the target to be warped

        Returns
        -------
        warping_index : 1D array
            warping index

        """
        if target not in ("query","reference"):
            raise ValueError("target argument must be 'query' or 'reference'")
        if target == "reference":
            xp = self.path[:,0]  # query path
            yp = self.path[:,1]  # reference path
        else:
            yp = self.path[:,0]  # query path
            xp = self.path[:,1]  # reference path
        interp_func = interp1d(xp,yp,kind="linear")
        warping_index = interp_func(np.arange(0,xp.max()+1)).astype(np.int64)
        return warping_index

    def plot_window(self):
        self._window.plot()

    def plot_cumsum_matrix(self):
        # extract max value with ignoring inf
        masked_array = np.ma.masked_array(self.cumsum_matrix,
            mask=self.cumsum_matrix == np.inf)
        _,ax = plt.subplots(1)
        sns.heatmap(self.cumsum_matrix.T,vmax=masked_array.max(),vmin=0,\
            xticklabels=self.cumsum_matrix.shape[0]//10,\
            yticklabels=self.cumsum_matrix.shape[1]//10,\
            ax=ax
        )
        ax.invert_yaxis()
        ax.set_xlabel("query index")
        ax.set_ylabel("reference index")
        ax.set_title("cumsum matrix")
        plt.show()

    def plot_path(self,with_=None):
        """plot alignment path

        Parameters
        ----------
        with_ : "win","cum" or None
            if given, following will be plotted with alignment path
            "win" : window matrix
            "cum" : cumsum matrix

        """
        if self.dist_only:
            raise Exception("alignment path not calculated.")
        _,ax = plt.subplots(1)
        if with_ is None:
            ax.plot(self.path[:,0],self.path[:,1])
        elif with_ == "win":
            sns.heatmap(self._window.matrix.T,vmin=0,vmax=1,\
                xticklabels=self._window.matrix.shape[0]//10,\
                yticklabels=self._window.matrix.shape[1]//10,\
                ax=ax
            )
            ax.plot(self.path[:,0],self.path[:,1])
            ax.invert_yaxis()
        elif with_ == "cum":
            # extract max value with ignoring inf
            masked_array = np.ma.masked_array(self.cumsum_matrix,
                mask=self.cumsum_matrix == np.inf)
            sns.heatmap(self.cumsum_matrix.T,vmax=masked_array.max(),vmin=0,\
                xticklabels=self.cumsum_matrix.shape[0]//10,\
                yticklabels=self.cumsum_matrix.shape[1]//10,\
                ax=ax
            )
            ax.plot(self.path[:,0],self.path[:,1])
            ax.invert_yaxis()
        else:
            raise NotImplementedError("'with_' argument only supports: 'win','cum'")
        ax.set_title("alignment path")
        ax.set_xlabel("query index")
        ax.set_ylabel("reference index")
        plt.show()


    def plot_pattern(self):
        self._pattern.plot()
