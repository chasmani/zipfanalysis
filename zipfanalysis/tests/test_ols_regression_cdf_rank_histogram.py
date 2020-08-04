from unittest import TestCase
import numpy as np

from zipfanalysis.estimators.ols_regression_cdf_rank_histogram import ols_regression_cdf_rank_histogram_estimator
from zipfanalysis.estimators.ols_regression_cdf_rank_histogram import get_survival_function_of_rank_histogram_points
from zipfanalysis.utilities.data_generators import get_counts_with_known_ranks

class TestSurvivalFunctionRankHistrogram(TestCase):

	def test_survival_function(self):

		ns = [10,5,2,2,1]
		ranks, sfs = get_survival_function_of_rank_histogram_points(ns)
		correct_sf = np.array([20,10,5,3,1])/sum(ns)
		np.testing.assert_almost_equal(sfs, correct_sf)
		np.testing.assert_almost_equal(ranks, [1,2,3,4,5])



	"""
	def test_with_generated_data(self):

		np.random.seed(2)
		for alpha in [0.4, 0.7, 1.1, 1.6, 1.9, 2.3, 2.5]:
			ns = get_counts_with_known_ranks(alpha, N=10000, W=15)
			alpha_result = estimate_ols_regression_cdf_rank_histogram(ns, min_frequency=4)
			self.assertAlmostEqual(alpha, alpha_result, places=1)

	"""
