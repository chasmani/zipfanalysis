

import time
from unittest import TestCase

import numpy as np

from zipfanalysis.estimators.clauset import clauset_maximise_likelihood, clauset_differential_root, powerlaw_package_clauset_estimator
from zipfanalysis.utilities.data_generators import get_counts_with_known_ranks, get_ranked_empirical_counts_from_infinite_power_law


class TestClausetEstimatorMatchesPackage(TestCase):

	def test_both_versions_match_powerlaw_package_simple_data(self):

		ns = [80,50,20,1,1,0]
		
		clauset_a = clauset_maximise_likelihood(ns)
		clauset_b = clauset_differential_root(ns)
		lib_alpha = powerlaw_package_clauset_estimator(ns)

		self.assertAlmostEqual(lib_alpha, clauset_a, places=4)
		self.assertAlmostEqual(lib_alpha, clauset_b, places=4)

	def test_both_version_match_powerlaw_package_generated_data(self):

		np.random.seed(1)
		for alpha in [1.4, 1.7, 2.1]:
			for N in [10, 100, 1000, 10000]:
				W = 1000

				ns = get_counts_with_known_ranks(alpha, N, W)

				clauset_a = clauset_maximise_likelihood(ns)
				clauset_b = clauset_differential_root(ns)
				lib_alpha = powerlaw_package_clauset_estimator(ns)

				self.assertAlmostEqual(lib_alpha, clauset_a, places=3)
				self.assertAlmostEqual(lib_alpha, clauset_b, places=3)


def speed_test():

	np.random.seed(1)
	ns = get_ranked_empirical_counts_from_infinite_power_law(1.1, 10000)

	start = time.time()
	for k in range(5):
		print(clauset_maximise_likelihood(ns))
	end = time.time()
	print("Clauset A is ", end - start)

	start = time.time()
	for k in range(5):
		print(clauset_differential_root(ns))
	end = time.time()
	print("Clauset B is ", end - start)

	start = time.time()
	for k in range(5):
		print(powerlaw_package_clauset_estimator(ns))
	end = time.time()
	print("Powerlaw package is ", end - start)

if __name__=="__main__":
	speed_test()