# -*- coding: utf-8 -*-
"""unittest
"""
import unittest
from numpy.testing import assert_almost_equal

from dtwalign import dtw
from dtwalign.dtw import _get_pattern
from dtwalign.step_pattern import *
from dtwalign.window import *
from test_data import gen_data
from rdtw import DtwR

import matplotlib.pyplot as plt

class TestDistance(unittest.TestCase):
    def setUp(self):
        self._gen_data()
        self._gen_pattern()
        self._gen_window()

    def tearDown(self):
        pass

    def _gen_data(self):
        """generate test data
        ex) open_begin=False,open_end=True
        self.X[0][1]
        """
        self.x = []
        self.y = []
        self.X = []

        for open_begin in [False,True]:
            tx = []
            ty = []
            tX = []
            for open_end in [False,True]:
                # x:query, y:reference, X:pair-wise cost matrix
                x,y,X = gen_data(open_begin,open_end)
                tx.append(x)
                ty.append(y)
                tX.append(X)

                # # plot data
                # plt.figure()
                # plt.plot(x,label="query");plt.plot(y,label="reference")
                # plt.legend()
                # plt.title("open_begin:{},open_end:{}".format(open_begin,open_end))
                # plt.show()

            self.x.append(tx)
            self.y.append(ty)
            self.X.append(tX)

    def _gen_pattern(self):
        self.patterns = [
            "symmetric1",
            "symmetric2",
            "symmetricP05",
            "symmetricP0",
            "symmetricP1",
            "symmetricP2",
            "asymmetric",
            "asymmetricP0",
            "asymmetricP05",
            "asymmetricP1",
            "asymmetricP2",
            "typeIa",
            "typeIb",
            "typeIas",
            "typeIbs",
            "typeIcs",
            "typeIds",
            "typeIIa",
            "typeIIb",
            "typeIIc",
            "typeIId",
            "typeIIIc",
            "typeIVc",
            "mori2006"
        ]

    def _gen_window(self):
        self.windows = dict(
            sakoechiba=20,
            itakura=20,
            none="none"
        )

    def _assert_dist(self,pattern,win_name,win_size,open_begin,open_end):
        if open_begin and win_name != "none":
            """
            Results with open-begin and using window differ between R and Python because
            R implementation doesn't consider zero-padded row separately
            (there should be no problem in practical use...)
            """
            pass
        elif (open_begin or open_end) and win_name == "itakura":
            """
            itakura window requires closed end
            """
            pass
        elif (open_begin or open_end) and not _get_pattern(pattern).is_normalizable:
            """
            partial matching requires normalizable step pattern
            """
            pass
        elif open_begin and _get_pattern(pattern).normalize_guide != "N":
            """
            open-begin matching requires "N"-normalizable step pattern
            """
            pass
        else:
            # get R result
            rdtw = DtwR(pattern,win_name,win_size,False,open_end,open_begin)
            rdtw.fit(self.X[int(open_begin)][int(open_end)])
            # get Python result
            pydtw = dtw(
                self.x[int(open_begin)][int(open_end)],
                self.y[int(open_begin)][int(open_end)],
                "euclidean",win_name,win_size,pattern,
                False,open_begin,open_end
            )
            # assert
            assert_almost_equal(rdtw.distance,pydtw.distance)

            if _get_pattern(pattern).is_normalizable:
                """
                only normalizable pattern can be asserted for normalization
                """
                assert_almost_equal(rdtw.normalized_distance,pydtw.normalized_distance)

    def test_asymmetric_distance(self):
        """asymmetric distance
        """
        asym_patterns = [pattern for pattern in self.patterns if pattern.find("asymmetric") == 0]
        for pattern in asym_patterns:
            for win_name,win_size in self.windows.items():
                for open_begin in [False,True]:
                    for open_end in [False,True]:
                        with self.subTest(pattern=pattern,win_name=win_name,\
                            win_size=win_size,open_begin=open_begin,open_end=open_end):
                            self._assert_dist(pattern,win_name,win_size,open_begin,open_end)

    def test_symmetric_distance(self):
        """symmetric distance
        """
        sym_patterns = [pattern for pattern in self.patterns if pattern.find("symmetric") == 0]
        # sym_patterns.remove("symmetric1")
        for pattern in sym_patterns:
            for win_name,win_size in self.windows.items():
                for open_begin in [False]:  # open-begin requires 'N' normalizable pattern
                    for open_end in [False,True]:
                        with self.subTest(pattern=pattern,win_name=win_name,\
                            win_size=win_size,open_begin=open_begin,open_end=open_end):
                            self._assert_dist(pattern,win_name,win_size,open_begin,open_end)

    def test_myers_distance(self):
        myers_patterns = [pattern for pattern in self.patterns if pattern.find("type") == 0]
        for pattern in myers_patterns:
            for win_name,win_size in self.windows.items():
                for open_begin in [False,True]:
                    for open_end in [False,True]:
                        with self.subTest(pattern=pattern,win_name=win_name,\
                            win_size=win_size,open_begin=open_begin,open_end=open_end):
                            self._assert_dist(pattern,win_name,win_size,open_begin,open_end)


if __name__ == "__main__":
    unittest.main(verbosity=2)
