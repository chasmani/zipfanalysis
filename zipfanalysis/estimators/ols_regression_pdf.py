
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

	if not all(isinstance(n, int) for n in ns):
		raise TypeError("The frequency count vector should be integers only. It should be the counts of words in order of most common")

	if not all(ns[i] >= ns[i+1] for i in range(len(ns)-1)):
		raise TypeError("The frequency count vector is not ordered correctly. It should be the counts of words in order of most common")		

	ns = np.array(ns)
	print(ns)
	ns_to_consider = ns[ns >= min_frequency]
	print(ns_to_consider)

	empirical_ranks = np.arange(1, len(ns_to_consider)+1)
	
	# Convert all the values to log space
	log_xs = np.log(empirical_ranks)
	log_ys = np.log(ns_to_consider)

	# This adds a constant so the regression is fit to y = ax + b
	log_xs = sm.add_constant(log_xs)

	# Fit a model to the logged data
	model = sm.OLS(log_ys, log_xs)
	results = model.fit()

	# Extract the slope from the regression results
	# Estimated exponents are based on p(x) = c k^(-lambda). 
	# To give lambda we multipl by -1 
	alpha = -1*results.params[1]
	print(results.params)
	return alpha

if __name__=="__main__":
	exp = ols_regression_pdf([5,4,3,2,1])
	print(exp)