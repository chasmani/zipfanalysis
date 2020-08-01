

from zipfanalysis.preprocessing.preprocessing import get_n_vector_from_text
from zipfanalysis.estimators.approximate_bayesian_computation import abc_regression_zipf

def abc_regression(book_filename):
	"""
	Takes a book as a filename
	Cleans the text, counts the words
	Applies approximate Bayesian computation on word counts to fit simple zipf model
	Return estimated exponent
	"""
	frequency_counts = get_n_vector_from_text(book_filename)
	alpha = abc_regression_zipf(frequency_counts)
	return alpha