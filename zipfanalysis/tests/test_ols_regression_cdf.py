

from unittest import TestCase
import numpy as np

from zipfanalysis.estimators.ols_regression_cdf import get_survival_function

class TestSurvivalFunction(TestCase):

	def test_survival_function(self):

		ns = [5,2,2,1]
		ranks, sfs = get_survival_function(ns)
		
		correct_sf = np.array([10,9,8,7,6,5,4,3,2,1])/sum(ns)
		np.testing.assert_almost_equal(sfs, correct_sf)
		
		correct_ranks = [1,1,1,1,1,2,2,3,3,4]
		np.testing.assert_almost_equal(ranks, correct_ranks)