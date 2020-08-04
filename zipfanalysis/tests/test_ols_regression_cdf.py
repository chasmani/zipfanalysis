

from unittest import TestCase
import numpy as np

from zipfanalysis.estimators.ols_regression_cdf import get_survival_function, ols_regression_cdf_estimator
from zipfanalysis.utilities.data_generators import get_ranked_empirical_counts_from_infinite_power_law

class TestSurvivalFunction(TestCase):

	def test_survival_function(self):

		ns = [5,2,2,1]
		ranks, sfs = get_survival_function(ns)
		
		correct_sf = np.array([10,9,8,7,6,5,4,3,2,1])/sum(ns)
		np.testing.assert_almost_equal(sfs, correct_sf)
		
		correct_ranks = [1,1,1,1,1,2,2,3,3,4]
		np.testing.assert_almost_equal(ranks, correct_ranks)


class TestOLSRegressionCDF(TestCase):

	def test_with_generated_data(self):
		"""
		The estimator isn't very accurate, but it should work for these exponents and parameters
		"""

		np.random.seed(2)
		for alpha in [1.2, 1.5, 1.7]:
			ns = get_ranked_empirical_counts_from_infinite_power_law(alpha, N=5000)
			alpha_result = ols_regression_cdf_estimator(ns, min_frequency=3)
			self.assertAlmostEqual(alpha, alpha_result, places=1)