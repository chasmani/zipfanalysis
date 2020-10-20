
import math
import sys

import numpy as np
import scipy
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from zipfanalysis.utilities.data_generators import get_ranked_empirical_counts_from_infinite_power_law
from zipfanalysis.utilities.data_generators import get_ranked_empirical_counts_from_infinite_zipf_mandelbrot_law


def abc_estimator(ns, min_exponent=1.01, max_exponent=2.2, trials_per_unit_of_exponent_range=1000):

	print("Running. This may take a few minutes . . . ")

	# Get enough samples to give a good resolution on the estimate
	n_samples = math.ceil((max_exponent - min_exponent) * trials_per_unit_of_exponent_range)

	# Calculate the target summary statistic
	target_summary_statistic = mean_log_of_observation_ranks(ns)

	# Total number of observations
	N = sum(ns) 
	print("Total words N is ", N)

	# Create a linspace of parameters
	test_parameters = np.linspace(min_exponent, max_exponent, n_samples)
	parameters = []
	distances = [] 

	print("")
	print("")
	print("Sampling from parameter space. . . ") 
	print("Min parameter = {} Max parameter = {}".format(min_exponent, max_exponent))

	# For each test parameter, generate data and measure its distance to the empirical data
	for test_param in test_parameters:

		sys.stdout.write('\r')
		percentage_complete = (test_param - min_exponent) / (max_exponent-min_exponent) * 100
		# the exact output you're looking for:
		bar_count_of_20 = int(percentage_complete/5)
		sys.stdout.write("[{}{}] test parameter = {:.2f}".format(bar_count_of_20*"=", (20-bar_count_of_20)*" ", test_param))
		sys.stdout.flush()

		test_ns = get_ranked_empirical_counts_from_infinite_power_law(test_param, N)
		print(test_param)
		kolmogorov_smirnov_distance(ns, test_ns)

		test_summary_statistic = mean_log_of_observation_ranks(test_ns)
		# Distance measure is the difference between summary statistics of test and empirical data sets 
		distance = test_summary_statistic - target_summary_statistic	

		parameters.append(test_param)
		distances.append(distance)
	print("")

	# Get trials that are "close" to the observed data
	successful_parameters, successful_distances = extract_successful_trials(parameters, distances)

	# Adjust the parameters along the regression line to approximate the posterior
	adjusted_params = regression_adjustment(successful_parameters, successful_distances)

	# Plot the kde and get the mle
	thetas, kde_data = sns.distplot(adjusted_params, bins=50).get_lines()[0].get_data()

	max_kde_index = np.argmax(kde_data)

	mle = thetas[max_kde_index]

	# If the given mle is close to the maximum exmained range, print a warning
	if mle > max_exponent - 0.1:
		print("WARNING - The maximum likelihood estimator is close to, or above, the upper bound on the range of investigated parameters of {}".format(max_exponent))
		print("We STRONGLY recommend you run the anlaysis again with a larger max_exponent")
		print("e.g. abc_regression_zipf(data, max_exponent={})".format(max_exponent+1))

	# Close the figure if it hasn't been shown - important
	plt.clf()
	return mle


def regression_adjustment(successful_parameters, successful_distances):
	"""
	Fit a regression model to the sucessful trials parameters and distance. 
	Then adjsut the parameters along the regression line to distance = 0
	"""
	parameters = np.array(successful_parameters)
	distances = np.array(successful_distances)
	distances_reshaped = distances.reshape((-1,1))

	# Fit a regression model
	model = LinearRegression()
	model.fit(distances_reshaped, parameters)
	
	beta_coef = model.coef_[0]
	alpha_intercept = model.intercept_

	# Adjust the parameters along the regression line to S_obs
	adjusted_params = parameters - beta_coef*distances
	return adjusted_params


def extract_successful_trials(parameters, distances, required_accepted_parameters=100):
	"""
	Get close trial results, at least as many as "required_accepted_parameters"
	"""
	# Choose a low tolerance to begin
	tolerance = 0.001
	finished = False
	# Keep on expanding the tolerance until we get enough successful trials
	while not finished:
		successes = np.absolute(distances) < tolerance
		success_count = np.count_nonzero(successes)
		if success_count<required_accepted_parameters:
			tolerance = tolerance*1.2
		else:
			finished = True

	successful_parameters = np.array(parameters)[successes]
	successful_distances = np.array(distances)[successes]
	return successful_parameters, successful_distances


def mean_log_of_observation_ranks(ns):
	"""
	Summary statistic for abc regression on power laws and zipfian distributions 
	The mean log sum of each observed event
	"""
	log_sum = 0
	for index in range(len(ns)):
		rank = index+1
		log_sum += ns[index] * ( np.log(rank))
	return log_sum/sum(ns)

def kolmogorov_smirnov_distance(ns_1, ns_2):
	"""
	Calculate the KS distance between two distributions
	"""

	xs_1 = []
	for i in range(1, len(ns_1)+1):
		xs_1 += [i]*ns_1[i-1]
	xs_2 = []
	for j in range(1, len(ns_2)+1):
		xs_2 += [j]*ns_2[j-1]

	test_1 = scipy.stats.ks_2samp(xs_1,xs_2)
	
	print("KS: ", test_1)
	test_2 = scipy.stats.wasserstein_distance(xs_1,xs_2)
	print("Wasserstein: ", test_2)

def abc_estimator_mandelbrot_zipf(ns, min_q=0, max_q=10, min_s=1.01, max_s=2, total_trials=1000):

	print("Running. This may take a few minutes . . . ")

	# Total number of observations
	N = sum(ns)
	print("Total words N is ", N)


	# Create a linspace of parameters
	Q, S = np.mgrid[min_q:max_q:1, min_s:max_s:0.1]
	qs = np.vstack((Q.flatten(), S.flatten())).T
	print(qs)

	parameters = []
	distances = [] 

	print("")
	print("")
	print("Sampling from parameter space. . . ") 

	# For each test parameter, generate data and measure its distance to the empirical data
	for test_params in qs:

		test_q = test_params[0]
		test_s = test_params[1]
		print(test_q, test_s)

		test_ns = get_ranked_empirical_counts_from_infinite_zipf_mandelbrot_law(test_s, test_q, N)
		
		kolmogorov_smirnov_distance(ns, test_ns)





def test_abc_mandelbrot():

	alpha = 1.1
	q = 5
	ns = get_ranked_empirical_counts_from_infinite_zipf_mandelbrot_law(alpha, q, N=5000)
	alpha_result = abc_estimator_mandelbrot_zipf(ns)




if __name__=="__main__":
	test_abc_mandelbrot()