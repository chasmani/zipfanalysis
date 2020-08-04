

from zipfanalysis.preprocessing.preprocessing import get_rank_frequency_from_text
from zipfanalysis.estimators.approximate_bayesian_computation import abc_estimator
from zipfanalysis.estimators.clauset import clauset_estimator
from zipfanalysis.estimators.ols_regression_cdf import ols_regression_cdf_estimator
from zipfanalysis.estimators.ols_regression_pdf import ols_regression_pdf_estimator


def abc(book_filename):
	"""
	Takes a book as a filename
	Cleans the text, counts the words
	Applies approximate Bayesian computation on word counts to fit simple zipf model
	Return estimated exponent
	"""
	frequency_counts = get_rank_frequency_from_text(book_filename)
	alpha = abc_estimator(frequency_counts)
	return alpha


def clauset(book_filename):
	"""
	Takes a book as a filename
	Cleans the text, counts the words
	Applies Clauset et al's estimator on word counts to fit simple zipf model
	Return estimated exponent
	"""
	frequency_counts = get_rank_frequency_from_text(book_filename)
	alpha = clauset_estimator(frequency_counts)
	return alpha	


def ols_cdf(book_filename, min_frequency=1):
	"""
	Takes a book as a filename
	Cleans the text, counts the words
	Applies estimator on word counts to fit simple zipf model
	Return estimated exponent
	"""
	frequency_counts = get_rank_frequency_from_text(book_filename)
	alpha = ols_regression_cdf_estimator(frequency_counts, min_frequency)
	return alpha		


def ols_pdf(book_filename, min_frequency=1):
	"""
	Takes a book as a filename
	Cleans the text, counts the words
	Applies estimator on word counts to fit simple zipf model
	Return estimated exponent
	"""
	frequency_counts = get_rank_frequency_from_text(book_filename)
	alpha = ols_regression_pdf_estimator(frequency_counts, min_frequency)
	return alpha