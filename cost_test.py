# -*- coding: utf-8 -*-

import unittest
from numpy.testing import assert_almost_equal

from pyfastdtw import dtw_low
from pyfastdtw.step_pattern import *
from pyfastdtw.window import *

from mtsa.common.dtw import DtwR

class TestDistance(unittest.TestCase):
    def setUp(self):
        # generate test data
        np.random.seed(12)
        self.X0 = np.ones([100,200])
        x1 = np.sin(2*np.pi*3*np.linspace(0,1,100))
        x1 += np.random.rand(x1.size)
        y1 = np.sin(2*np.pi*3.1*np.linspace(0,1,101))
        y1 += np.random.rand(y1.size)
        self.X1 = x1[:,np.newaxis].dot(y1[np.newaxis,:])
        # self.X0 = self.X1
        self.X2 = self.X1

    def tearDown(self):
        pass

    def test_distance_symmetric2(self):
        # R
        r0 = DtwR(step_pattern="symmetric2")
        r0.fit(self.X0)
        r1 = DtwR(step_pattern="symmetric2")
        r1.fit(self.X1)
        r2 = DtwR(step_pattern="symmetric2")
        r2.fit(self.X2)
        # pyfastdtw
        sym2 = Symmetric2()
        w0 = NoWindow(self.X0.shape[0],self.X0.shape[1])
        py0 = dtw_low(self.X0,w0,pattern=sym2)
        w1 = NoWindow(self.X1.shape[0],self.X1.shape[1])
        py1 = dtw_low(self.X1,w1,pattern=sym2)
        w2 = NoWindow(self.X2.shape[0],self.X2.shape[1])
        py2 = dtw_low(self.X2,w2,pattern=sym2)
        #assert
        assert_almost_equal(r0.distance,py0.distance)
        assert_almost_equal(r1.distance,py1.distance)
        assert_almost_equal(r2.distance,py2.distance)
        assert_almost_equal(r0.normalized_distance,py0.normalized_distance)
        assert_almost_equal(r1.normalized_distance,py1.normalized_distance)
        assert_almost_equal(r2.normalized_distance,py2.normalized_distance)

    def test_distance_symmetricP2(self):
        # pyfastdtw
        symp2 = SymmetricP2()
        w0 = NoWindow(self.X0.shape[0],self.X0.shape[1])
        py0 = dtw_low(self.X0,w0,pattern=symp2)
        w1 = NoWindow(self.X1.shape[0],self.X1.shape[1])
        py1 = dtw_low(self.X1,w1,pattern=symp2)
        w2 = NoWindow(self.X2.shape[0],self.X2.shape[1])
        py2 = dtw_low(self.X2,w2,pattern=symp2)

        # R
        r0 = DtwR(step_pattern="symmetricP2",distance_only=True)
        r0.fit(self.X0)
        r1 = DtwR(step_pattern="symmetricP2",distance_only=True)
        r1.fit(self.X1)
        r2 = DtwR(step_pattern="symmetricP2",distance_only=True)
        r2.fit(self.X2)

        #assert
        assert_almost_equal(r0.distance,py0.distance)
        assert_almost_equal(r1.distance,py1.distance)
        assert_almost_equal(r2.distance,py2.distance)
        assert_almost_equal(r0.normalized_distance,py0.normalized_distance)
        assert_almost_equal(r1.normalized_distance,py1.normalized_distance)
        assert_almost_equal(r2.normalized_distance,py2.normalized_distance)

if __name__ == "__main__":
    unittest.main(verbosity=2)
