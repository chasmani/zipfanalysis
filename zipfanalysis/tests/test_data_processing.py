
import unittest
from unittest import TestCase
import numpy as np

import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import src.data_processing as data_processing


class TestDataProcessing(TestCase):

	def test_reading_file(self):
		"""
		"""
		test_filepath = "../data/test_data/basic_test_data.txt"
		text = data_processing.read_txt_file(test_filepath)

		self.assertEqual(text, "Hello hello hello there this is a test file, what's it to you? Is it a big deal. Hello? ? 32.123")

		clean_text = data_processing.clean_text(text)

		self.assertEqual(clean_text, "Hello hello hello there this is a test file, whats it to you? Is it a big deal. Hello? ? 32.123")

		word_list = data_processing.convert_text_to_word_list(clean_text)
		
		self.assertEqual(word_list, ['hello', 'hello', 'hello', 'there', 'this', 'is', 'a', 'test', 'file', 'whats', 'it', 'to', 'you', 'is', 'it', 'a', 'big', 'deal', 'hello'])

		print(word_list)

		frequency_dist = data_processing.convert_word_list_to_frequency_dist(word_list)

		self.assertEqual(frequency_dist, [4, 2, 2, 2,1,1,1,1,1,1,1,1,1])

		frequency_counts = data_processing.convert_frequency_dist_to_frequency_counts(frequency_dist)

		self.assertEqual(frequency_counts, [9, 3, 0, 1])

		frequency_dist_2 = data_processing.convert_frequency_counts_to_frequency_dist(frequency_counts)

		self.assertEqual(frequency_dist, frequency_dist_2)



if __name__ == '__main__':
    unittest.main()
