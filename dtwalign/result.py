# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

class DtwResult():
    """Result of DTW.
    
    Attributes
    ----------
    path : 2d array
        Alignment path.  
        * First column: query path array
        * Second column: reference path array
    distance : float
        Alignment distance.
    normalized_distance : float
        Normalized alignment distance.

    """
    def __init__(self, cumsum_matrix, path, window, pattern):
        self.cumsum_matrix = cumsum_matrix

        if path is None:
            self.dist_only = True
        else:
            self.dist_only = False
            self.path = path

        self._window = window
        self._pattern = pattern

    def get_warping_path(self, target="query"):
        """Get warping path.

        Parameters
        ----------
        target : string, "query" or "reference"
            Specify the target to be warped.

        Returns
        -------
        warping_index : 1D array
            Warping index.

        """
        if target not in ("query", "reference"):
            raise ValueError("target argument must be 'query' or 'reference'")
        if target == "reference":
            xp = self.path[:, 0]  # query path
            yp = self.path[:, 1]  # reference path
        else:
            yp = self.path[:, 0]  # query path
            xp = self.path[:, 1]  # reference path
        interp_func = interp1d(xp, yp, kind="linear")
        # get warping index as float values and then convert to int
        # note: Ideally, the warped value should be calculated as mean.
        #       (in this implementation, just use value corresponds to rounded-up index)
        warping_index = interp_func(np.arange(xp.min(), xp.max()+1)).astype(np.int64)
        # the most left side gives nan, so substitute first index of path
        warping_index[0] = yp.min()

        return warping_index

    def plot_window(self):
        self._window.plot()

    def plot_cumsum_matrix(self):
        # extract max value with ignoring inf
        masked_array = np.ma.masked_array(self.cumsum_matrix,
            mask=self.cumsum_matrix == np.inf)
        _,ax = plt.subplots(1)
        sns.heatmap(self.cumsum_matrix.T, vmax=masked_array.max(), vmin=0,
            xticklabels=self.cumsum_matrix.shape[0]//10,
            yticklabels=self.cumsum_matrix.shape[1]//10,
            ax=ax
        )
        ax.invert_yaxis()
        ax.set_xlabel("query index")
        ax.set_ylabel("reference index")
        ax.set_title("cumsum matrix")
        plt.show()

    def plot_path(self, with_="cum"):
        """Plot alignment path.

        Parameters
        ----------
        with_ : string, "win" or "cum" or None
            If given, following will be plotted with alignment path:  
            * "win" : window matrix
            * "cum" : cumsum matrix

        """
        if self.dist_only:
            raise Exception("alignment path not calculated.")
        _, ax = plt.subplots(1)
        if with_ is None:
            ax.plot(self.path[:, 0], self.path[:, 1])
        elif with_ == "win":
            sns.heatmap(self._window.matrix.T, vmin=0, vmax=1,
                xticklabels=self._window.matrix.shape[0]//10,
                yticklabels=self._window.matrix.shape[1]//10,
                ax=ax)
            ax.plot(self.path[:, 0], self.path[:, 1], "b")
            ax.invert_yaxis()
        elif with_ == "cum":
            # extract max value with ignoring inf
            masked_array = np.ma.masked_array(self.cumsum_matrix,
                mask=self.cumsum_matrix == np.inf)
            sns.heatmap(self.cumsum_matrix.T, vmax=masked_array.max(), vmin=0,
                xticklabels=self.cumsum_matrix.shape[0]//10,
                yticklabels=self.cumsum_matrix.shape[1]//10,
                ax=ax)
            ax.plot(self.path[:, 0], self.path[:, 1], "y")
            ax.invert_yaxis()
        else:
            raise NotImplementedError("'with_' argument only supports: 'win','cum'")
        ax.set_title("alignment path")
        ax.set_xlabel("query index")
        ax.set_ylabel("reference index")
        plt.show()

    def plot_pattern(self):
        self._pattern.plot()
