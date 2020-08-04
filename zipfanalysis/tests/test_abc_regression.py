

from unittest import TestCase
import numpy as np

from zipfanalysis.estimators.approximate_bayesian_computation import abc_estimator
from zipfanalysis.utilities.data_generators import get_ranked_empirical_counts_from_infinite_power_law


class TestABCRegression(TestCase):

	def test_with_generated_data(self):
		"""
		"""

		np.random.seed(2)
		for alpha in [1.01, 1.3, 1.5, 1.7, 2.1]:
			ns = get_ranked_empirical_counts_from_infinite_power_law(alpha, N=5000)
			alpha_result = abc_estimator(ns, max_exponent=alpha+0.2)
			self.assertAlmostEqual(alpha, alpha_result, places=1)