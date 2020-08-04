
import numpy as np
import statsmodels.api as sm


def get_survival_function_of_rank_histogram_points(ns, min_frequency = 1):
	"""
	Get the survival function of P(X>r_e), r_e is empirical ranks
	Consider the points in the values of the frequency counts vector, n[r_e] is the count of the (r_e)th most common word 
	That is technically a histogram
	Only works with decreasing counts i.e. rank frequency data
	"""
	ranks = []
	sfs = []
	# Iterate through ranks
	for r in range(1, len(ns)+1):
		# Only consider ranks with counts greater than or equal to the low frequency cutoff
		if ns[r-1] >= min_frequency:
			ranks.append(r)
			# Count total words with ranks above or equal to this rank  
			cs = sum(ns[r-1:])
			sfs.append(cs)

	# Normalise the survival function based on the entire frequency counts vector (including low frequency cut offs)
	sfs = np.array(sfs)/sum(ns)
	return ranks, sfs


def ols_regression_cdf_rank_histogram(ns, min_frequency = 1):
	"""
	Perform regression on the CDF of the frequency of words
	Assume model of the form p(r_e) = c r_e(-alpha)
	r_e is empirical rank of a word
	CDF would be P(r_e) = c r_e(-alpha+1)
	We actually use the survival function 1-CDF, also known as the complementary CDF
	"""
	# Validate that the frequency counts vector is of the right form
	if not all(isinstance(n, int) for n in ns):
		raise TypeError("The frequency count vector should be integers only. It should be the counts of words in order of most common")

	ns = np.array(ns)

	# Get the survival function, with 1 point for every rank based on the frequency count vector n[r_e]
	ranks, sfs = get_survival_function_of_rank_histogram_points(ns, min_frequency)
	
	# Convert all the values to log space
	log_xs = np.log(ranks)
	log_ys = np.log(sfs)

	# This adds a constant so the regression is fit to y = ax + b
	log_xs = sm.add_constant(log_xs)

	# Fit a model to the logged data
	model = sm.OLS(log_ys, log_xs)
	results = model.fit()

	# Return the regression results 
	c = results.params[0]
	lamb_hat = results.params[1]
	return c, lamb_hat


def ols_regression_cdf_rank_histogram_estimator(ns, min_frequency = 1):
	min_frequency = max(1, min_frequency)
	c, lamb_hat = ols_regression_cdf_rank_histogram(ns, min_frequency)
	print(c, lamb_hat)
	alpha_estimate = 1 - lamb_hat
	return alpha_estimate
