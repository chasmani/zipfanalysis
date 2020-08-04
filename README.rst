============
zipfanalysis
============

Tools in python for analysing Zipf's law from text samples. 

This can be installed as a package from the python3 package library using the terminal command:
::

	>>> pip install zipfanalysis

-----
Usage
-----

The package can be used from within python scripts to estimate Zipf exponents, assuming a simple power law model for 
word frequencies and ranks. To use the pacakge import it using
::

	import zipfanalysis

-------------
Simple Method
-------------

The easiest way to carry out an analysis on a book or text file, using different estimators, is:
::

	alpha_clauset = zipfanalysis.clauset("path_to_book.txt")

	alpha_pdf = zipfanalysis.ols_pdf("path_to_book.txt", min_frequency=3)

	alpha_cdf = zipfanalysis.ols_cdf("path_to_book.txt", min_frequency=3)

	alpha_abc = zipfanalysis.abc("path_to_book.txt")

---------------
In Depth Method
---------------

Convert a book or text file to the frequency of words, ranked from highest to lowest: 
::

	word_counts = zipfanalysis.preprocessing.preprocessing.get_rank_frequency_from_text("path_to_book.txt")
	

Carry out different types of analysis to fit a power law to the data:
::

	# Clauset et al estimator
	alpha_clauset = zipfanalysis.estimators.clauset.clauset_estimator(word_counts)

	# Ordinary Least Squares regression on log(rank) ~ log(frequency) 
	# Optional low frequency cut-off
	alpha_pdf = zipfanalysis.estimators.ols_regression_pdf.ols_regression_pdf_estimator(word_counts, min_frequency=2)

	# Ordinary least squares regression on the complemantary cumulative distribution function of ranks
	# OLS on log(P(R>rank)) ~ log(rank) 
	# Optional low frequency cut-off 
	alpha_cdf = zipfanalysis.estimators.ols_regression_cdf.ols_regression_cdf_estimator(word_counts)

	# Approximate Bayesian computation (regression method)
	# Assumes model of p(rank) = C prob_rank^(-alpha)
	# prob_rank is a word's rank in an underlying probability distribution
	alpha_abc = zipfanalysis.estimators.approximate_bayesian_computation.abc_estimator(word_counts)
