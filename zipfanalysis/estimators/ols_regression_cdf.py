
import numpy as np
import statsmodels.api as sm

def get_survival_function(ns, f_min = 1):
	"""
	Get the surivival function, now with points for each xs
	Don't get points with a low frequency cut off (but still consider their effect in terms of the other points)
	"""

	ranks = []
	sfs = []
	# Iterate through ranks
	for r in range(1, len(ns)+1):
		# Only consider ranks with counts greater than or equal to the low frequency cutoff
		if ns[r-1] >= f_min:
			# Add the rank and sf for each data point - not just totals for each rank
			# So 1 point for 1 token in a book. 
			# If the word "the" appears 1000 times then it is represented by 1000 points

			for x in range(ns[r-1]):

				ranks.append(r)
				cs = sum(ns[r-1:]) - x
				sfs.append(cs)

	# Normalise based on full size of ns
	sfs = np.array(sfs)/sum(ns)

	return ranks, sfs


def ols_regression_cdf(ns, min_frequency = 1):
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
	ranks, sfs = get_survival_function(ns, min_frequency)
	
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


def ols_regression_cdf_estimator(ns, min_frequency = 1):
	min_frequency = max(1, min_frequency)
	c, lamb_hat = ols_regression_cdf(ns, min_frequency)
	alpha_estimate = 1 - lamb_hat
	return alpha_estimate