from unittest import TestCase
import numpy as np

from zipfanalysis.estimators.ols_regression_pdf import ols_regression_pdf

class TestOLSRegressionPDF(TestCase):
	

	def test_with_known_exponents(self):

		for alpha in [0.1, 0.5, 0.8, 1.1, 1.6, 1.9, 2.2, 2.5, 3.1]:
			ranks = np.arange(1,10)
			ns = [int((r**(-1*alpha))*10000000) for r in ranks]
			alpha_result = ols_regression_pdf(ns)
			self.assertAlmostEqual(alpha, alpha_result, places=4)

	def test_with_zeros(self):

		for alpha in [0.1, 0.5, 0.8, 1.1, 1.6, 1.9, 2.2, 2.5, 3.1]:
			ranks = np.arange(1,10)
			ns = [int((r**(-1*alpha))*10000000) for r in ranks]
			ns += [0,0,0,0,0]
			alpha_result = ols_regression_pdf(ns)
			self.assertAlmostEqual(alpha, alpha_result, places=4)
		
	def test_with_low_frequency_cutoff(self):

		for alpha in [0.1, 0.5, 0.8, 1.1, 1.6, 1.9, 2.2, 2.5, 3.1]:
			ranks = np.arange(1,10)
			ns = [int((r**(-1*alpha))*10000000) for r in ranks]
			ns += [1]*20
			alpha_result = ols_regression_pdf(ns, min_frequency=2)
			self.assertAlmostEqual(alpha, alpha_result, places=4)

	def test_with_low_frequency_cutoff_b(self):

		for alpha in [0.1, 0.5, 0.8, 1.1, 1.6, 1.9, 2.2, 2.5, 3.1]:
			ranks = np.arange(1,10)
			ns = [int((r**(-1*alpha))*10000000) for r in ranks]
			ns += [2]*20 + [1]*20
			print(ns)
			alpha_result = ols_regression_pdf(ns, min_frequency=3)
			self.assertAlmostEqual(alpha, alpha_result, places=4)

		
	def test_with_floats(self):

		ns = [6,4,3,2.1,1,1]
		with self.assertRaises(TypeError):
			ols_regression_pdf(ns)

	def test_with_words(self):

		ns = [6,4,3,"Bob",1,1]
		with self.assertRaises(TypeError):
			ols_regression_pdf(ns)

	def test_with_unordered_vector(self):

		ns = [1,2,3,4]
		with self.assertRaises(TypeError):
			ols_regression_pdf(ns)		
		