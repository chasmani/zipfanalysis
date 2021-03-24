
import unittest
from unittest import TestCase
import numpy as np

import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import src.clauset as clauset
from main import ZipfAnalysis

class TestClauset(TestCase):

	def test_against_packages_dummy_data(self):
		"""
		"""
		ns = [21, 6, 2, 1, 1]
		plaw_fit = clauset.powerlaw_package_clauset_estimator(ns)
		zipfanalysis_fit = clauset.clauset_estimator(ns)
		zipfanalysis_max_l_fit =  clauset.clauset_maximise_likelihood(ns)
		self.assertAlmostEqual(plaw_fit, zipfanalysis_fit, 3)
		self.assertAlmostEqual(plaw_fit, zipfanalysis_max_l_fit, 3)
		self.assertAlmostEqual(zipfanalysis_max_l_fit, zipfanalysis_fit, 3)

		ns = [11, 9, 5, 4, 4,3,3,3,2,1,1]
		plaw_fit = clauset.powerlaw_package_clauset_estimator(ns)
		zipfanalysis_fit = clauset.clauset_estimator(ns)
		zipfanalysis_max_l_fit =  clauset.clauset_maximise_likelihood(ns)
		self.assertAlmostEqual(plaw_fit, zipfanalysis_fit, 3)
		self.assertAlmostEqual(plaw_fit, zipfanalysis_max_l_fit, 3)
		self.assertAlmostEqual(zipfanalysis_max_l_fit, zipfanalysis_fit, 3)

	def test_against_books(self):
		"""
		Test the zipfanalysis estimator to ensure the same results
		against the powerlaw package and plfit package
		"""
		# Commented out longer books to keep tests quick
		# Tests pass with longer books too
		books = [
			"alice_wonderland.txt",
			#"chronicle_london.txt",
			#"moby_dick.txt",
			#"tale_two_cities.txt",
			#"ulysses.txt"
		]

		for book in books:
			print("Testing ", book)
			book_filepath = "../data/books/{}".format(book)
			za = ZipfAnalysis(data=book_filepath, data_type="input_filename")
			ns = za.frequency_dist
			plaw_fit = clauset.powerlaw_package_clauset_estimator(ns)
			zipfanalysis_fit = clauset.clauset_estimator(ns)
			zipfanalysis_max_l_fit =  clauset.clauset_maximise_likelihood(ns)
			print(zipfanalysis_fit)
			self.assertAlmostEqual(plaw_fit, zipfanalysis_fit, 3)
			self.assertAlmostEqual(plaw_fit, zipfanalysis_max_l_fit, 3)
			self.assertAlmostEqual(zipfanalysis_max_l_fit, zipfanalysis_fit, 3)			


if __name__ == '__main__':
    unittest.main()
