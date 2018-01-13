# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def _num_to_str(num):
    if type(num) == int:
        return str(num)
    elif type(num) == float:
        return "{0:1.2f}".format(num)
    else:
        return str(num)

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

    @property
    def is_normalizable(self):
        return self.normalize_guide != "none"

    def plot(self):
        plt.figure(figsize=(6,6))
        if not hasattr(self,"_graph"):
            self._gen_graph()
        nx.draw_networkx_nodes(self._graph,\
            pos=self._graph_layout)
        nx.draw_networkx_edges(self._graph,\
            pos=self._graph_layout)
        nx.draw_networkx_edge_labels(self._graph,\
            pos=self._graph_layout,
            edge_labels=self._edge_labels)
        min_index = min([min(pat["indices"][0]) for pat in self.pattern_info])
        plt.xlim([min_index - 0.5,0.5])
        plt.ylim([min_index - 0.5,0.5])
        plt.title(self.label + str(" pattern"))
        plt.xlabel("query index")
        plt.ylabel("reference index")
        plt.show()

    def _normalize(self,row,n,m):
        """normalize

        Parameters
        ----------
        row : 1D array
            expect last row of D
        n : int
            length of query (D.shape[0])
        m : int
            length of reference (D.shape[1])
        """
        if not self.is_normalizable:
            return None
        if self.normalize_guide == "N+M":
            return row/(n + np.arange(1,m+1))
        elif self.normalize_guide == "N":
            return row/n
        elif self.normalize_guide == "M":
            return row/np.arange(1,m+1)
        else:
            raise Exception()

    def _gen_graph(self):
        graph = nx.DiGraph()
        graph_layout = dict()
        edge_labels = dict()
        node_names = []
        # set node
        for pidx,pat in enumerate(self.pattern_info):
            step_len = len(pat["indices"])
            nn = []
            for sidx in range(step_len):
                node_name = str(pidx) + str(sidx)
                graph.add_node(node_name)
                graph_layout[node_name] = \
                    np.array(pat["indices"][sidx])
                nn.append(node_name)
            node_names.append(nn)
        # set edge
        for pidx,pat in enumerate(self.pattern_info):
            step_len = len(pat["indices"])
            for sidx in range(step_len-1):
                graph.add_edge(node_names[pidx][sidx],
                    node_names[pidx][sidx+1])
                edge_labels[(node_names[pidx][sidx],
                    node_names[pidx][sidx+1])] = _num_to_str(pat["weights"][sidx])
        self._graph = graph
        self._graph_layout = graph_layout
        self._edge_labels = edge_labels

    def _get_array(self):
        """
        get pattern information as np.ndarray for numba.jit
        """
        array = np.zeros([self.num_pattern,self.max_pattern_len,3],dtype="float")
        for pidx in range(self.num_pattern):
            pattern_len = len(self.pattern_info[pidx]["indices"])
            for sidx in range(pattern_len):
                array[pidx,sidx,0:2] = self.pattern_info[pidx]["indices"][sidx]
                if sidx == 0:
                    array[pidx,sidx,2] = np.nan
                else:
                    array[pidx,sidx,2] = self.pattern_info[pidx]["weights"][sidx-1]
        self.array = array

    def __repr__(self):
        p_info = self.pattern_info
        rv = self.label + " pattern: \n\n"
        for pidx in range(self.num_pattern):
            rv += "pattern " + str(pidx) + ": "
            pattern_len = len(p_info[pidx]["indices"])
            p_str = str(p_info[pidx]["indices"][0])
            for sidx in range(1,pattern_len):
                p_str += " - ["
                p_str += _num_to_str(p_info[pidx]["weights"][sidx-1])
                p_str += "] - "
                p_str += str(p_info[pidx]["indices"][sidx])
            rv += p_str + "\n"
        rv += "\nnormalization guide: " + self.normalize_guide
        return rv

class Symmetric1(BasePattern):
    label = "symmetric1"
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
    label = "symmetric2"
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

class SymmetricP0(Symmetric2):
    """same as symmetric2 pattern"""
    label = "symmetricP05"

class SymmetricP05(BasePattern):
    label = "symmetricP05"
    pattern_info = [
        dict(
            indices=[(-1,-3),(0,-2),(0,-1),(0,0)],
            weights=[2,1,1]
        ),
        dict(
            indices=[(-1,-2),(0,-1),(0,0)],
            weights=[2,1]
        ),
        dict(
            indices=[(-1,-1),(0,0)],
            weights=[2]
        ),
        dict(
            indices=[(-2,-1),(-1,0),(0,0)],
            weights=[2,1]
        ),
        dict(
            indices=[(-3,-1),(-2,0),(-1,0),(0,0)],
            weights=[2,1,1]
        )
    ]
    normalize_guide = "N+M"

    def __init__(self):
        super().__init__()

class SymmetricP1(BasePattern):
    label = "symmetricP1"
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
    label = "symmetricP2"
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


class Asymmetric(BasePattern):
    label = "asymmetric"
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
            indices=[(-1,-2),(0,0)],
            weights=[1]
        )
    ]
    normalize_guide = "N"

    def __init__(self):
        super().__init__()

class AsymmetricP0(BasePattern):
    label = "asymmetricP0"
    pattern_info = [
        dict(
            indices=[(0,-1),(0,0)],
            weights=[0]
        ),
        dict(
            indices=[(-1,-1),(0,0)],
            weights=[1]
        ),
        dict(
            indices=[(-1,0),(0,0)],
            weights=[1]
        )
    ]
    normalize_guide = "N"

    def __init__(self):
        super().__init__()

class AsymmetricP05(BasePattern):
    label = "asymmetricP05"
    pattern_info = [
        dict(
            indices=[(-1,-3),(0,-2),(0,-1),(0,0)],
            weights=[1/3,1/3,1/3]
        ),
        dict(
            indices=[(-1,-2),(0,-1),(0,0)],
            weights=[0.5,0.5]
        ),
        dict(
            indices=[(-1,-1),(0,0)],
            weights=[1]
        ),
        dict(
            indices=[(-2,-1),(-1,0),(0,0)],
            weights=[1,1]
        ),
        dict(
            indices=[(-3,-1),(-2,0),(-1,0),(0,0)],
            weights=[1,1,1]
        )
    ]
    normalize_guide = "N"

    def __init__(self):
        super().__init__()

class AsymmetricP1(BasePattern):
    label = "asymmetricP1"
    pattern_info = [
        dict(
            indices=[(-1,-2),(0,-1),(0,0)],
            weights=[0.5,0.5]
        ),
        dict(
            indices=[(-1,-1),(0,0)],
            weights=[1]
        ),
        dict(
            indices=[(-2,-1),(-1,0),(0,0)],
            weights=[1,1]
        )
    ]
    normalize_guide = "N"

    def __init__(self):
        super().__init__()

class AsymmetricP2(BasePattern):
    label = "asymmetricP2"
    pattern_info = [
        dict(
            indices=[(-2,-3),(-1,-2),(0,-1),(0,0)],
            weights=[2/3,2/3,2/3]
        ),
        dict(
            indices=[(-1,-1),(0,0)],
            weights=[1]
        ),
        dict(
            indices=[(-3,-2),(-2,-1),(-1,0),(0,0)],
            weights=[1,1,1]
        )
    ]
    normalize_guide = "N"

    def __init__(self):
        super().__init__()
