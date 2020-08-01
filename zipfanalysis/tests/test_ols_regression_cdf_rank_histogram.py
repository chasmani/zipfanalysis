from unittest import TestCase
import numpy as np

from zipfanalysis.estimators.ols_regression_cdf_rank_histogram import estimate_ols_regression_cdf_rank_histogram
from zipfanalysis.utilities.data_generators import get_counts_with_known_ranks

class TestOLSRegressionCDFRanksHistogram(TestCase):

	"""
	def test_with_known_exponents(self):

		for alpha in [1.1, 1.6, 1.9, 2.2, 2.5, 3.1]:
			ranks = np.arange(1,10)
			ns = [int((r**(-1*alpha))*10000000) for r in ranks]
			alpha_result = estimate_ols_regression_cdf_rank_histogram(ns)
			self.assertAlmostEqual(alpha, alpha_result, places=4)

	def test_with_zeros(self):

		for alpha in [1.1, 1.6, 1.9, 2.2, 2.5, 3.1]:
			ranks = np.arange(1,10)
			ns = [int((r**(-1*alpha))*10000000) for r in ranks]
			ns += [0,0,0,0,0]
			alpha_result = estimate_ols_regression_cdf_rank_histogram(ns)
			self.assertAlmostEqual(alpha, alpha_result, places=4)
		
	def test_with_low_frequency_cutoff(self):

		for alpha in [1.1, 1.6, 1.9, 2.2, 2.5, 3.1]:
			ranks = np.arange(1,10)
			ns = [int((r**(-1*alpha))*10000000) for r in ranks]
			ns += [1]*20
			alpha_result = estimate_ols_regression_cdf_rank_histogram(ns, min_frequency=2)
			self.assertAlmostEqual(alpha, alpha_result, places=4)

	def test_with_low_frequency_cutoff_b(self):

		for alpha in [1.1, 1.6, 1.9, 2.2, 2.5, 3.1]:
			ranks = np.arange(1,10)
			ns = [int((r**(-1*alpha))*10000000) for r in ranks]
			ns += [2]*20 + [1]*20
			print(ns)
			alpha_result = estimate_ols_regression_cdf_rank_histogram(ns, min_frequency=3)
			self.assertAlmostEqual(alpha, alpha_result, places=4)
		
	def test_with_floats(self):

		ns = [6,4,3,2.1,1,1]
		with self.assertRaises(TypeError):
			estimate_ols_regression_cdf_rank_histogram(ns)

	def test_with_words(self):

		ns = [6,4,3,"Bob",1,1]
		with self.assertRaises(TypeError):
			estimate_ols_regression_cdf_rank_histogram(ns)

	def test_with_unordered_vector(self):

		ns = [1,2,3,4]
		with self.assertRaises(TypeError):
			estimate_ols_regression_cdf_rank_histogram(ns)		
	"""
	def test_with_generated_data(self):

		np.random.seed(2)
		for alpha in [0.4, 0.7, 1.1, 1.6, 1.9, 2.3, 2.5]:
			ns = get_counts_with_known_ranks(alpha, N=10000, W=15)
			alpha_result = estimate_ols_regression_cdf_rank_histogram(ns, min_frequency=4)
			self.assertAlmostEqual(alpha, alpha_result, places=1)


