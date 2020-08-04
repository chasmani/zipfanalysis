
import math
import sys

import numpy as np
import scipy
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from zipfanalysis.utilities.data_generators import get_ranked_empirical_counts_from_infinite_power_law



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


if __name__=="__main__":
	alpha = 1.1
	ns = get_ranked_empirical_counts_from_infinite_power_law(alpha, N=5000)
	alpha_result = abc_regression_zipf(ns, max_exponent=1.2)
	print(alpha_result)