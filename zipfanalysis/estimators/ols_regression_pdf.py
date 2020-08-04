
import numpy as np
import statsmodels.api as sm


def ols_regression_pdf(ns, min_frequency=1):
	"""
	Ordinary Least Squares regression on the empirical PDF
	Input is a frequency count vector, n[k] is the count of the kth most common word
	Linear regression in log space, based on y = log(n), x= log(k)
	Assumes a model of the form p(k) = c k ^(-alpha)
	Return alpha 
	"""

	# Validate that the frequency counts vector is of the right form
	if not all(isinstance(n, int) for n in ns):
		raise TypeError("The frequency count vector should be integers only. It should be the counts of words in order of most common")

	ns = np.array(ns)
	# Cut off low frequency points
	ns_to_consider = ns[ns >= min_frequency]

	empirical_ranks = np.arange(1, len(ns_to_consider)+1)
	
	# Convert all the values to log space
	log_xs = np.log(empirical_ranks)
	log_ys = np.log(ns_to_consider)

	# This adds a constant so the regression is fit to y = ax + b
	log_xs = sm.add_constant(log_xs)

	# Fit a model to the logged data
	model = sm.OLS(log_ys, log_xs)
	results = model.fit()

	# Return the regression results 
	c = results.params[0]
	lamb_hat = results.params[1]
	return c, lamb_hat

def ols_regression_pdf_estimator(ns, min_frequency = 1):
	min_frequency = max(1, min_frequency)
	c, lamb_hat = ols_regression_pdf(ns, min_frequency)
	# To give alpha we multiply by -1 
	alpha_estimate = -1 * lamb_hat
	return alpha_estimate