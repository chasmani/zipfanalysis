
import numpy as np
import matplotlib.pyplot as plt

from zipfanalysis.utilities.data_generators import get_counts_with_known_ranks, get_ranked_empirical_counts_from_infinite_power_law
from zipfanalysis.estimators.ols_regression_cdf import get_survival_function, ols_regression_cdf
from zipfanalysis.estimators.ols_regression_cdf_rank_histogram import get_survival_function_of_rank_histogram_points, ols_regression_cdf_rank_histogram

def plot_sf_of_known_exponent_rank_histogram():

	np.random.seed(3)
	alpha = 2.5
	ns = get_ranked_empirical_counts_from_infinite_power_law(alpha, N=1000)
	print(ns)
	ranks, sfs = get_survival_function_of_rank_histogram_points(ns)
	# Plot the empirical sf
	plt.scatter(np.array(ranks), sfs, label="Empirical SF")

	sf_alpha = -1*(alpha-1)

	# Plot the predicted sf, based on alpha
	c = 0.1
	sf_hats = []
	for r in ranks:
		f = np.exp(c)*r**sf_alpha
		sf_hats.append(f)
	plt.plot(ranks, sf_hats, label="Expected SF")

	# Plot the estimated sf
	c, lamb_hat = ols_regression_cdf_rank_histogram(ns, min_frequency=3)
	sf_hats = []
	for r in ranks:
		f = np.exp(c)*r**lamb_hat
		sf_hats.append(f)
	plt.plot(ranks, sf_hats, label="Estimated")	
	

	print("Expected empiricial sf exponent is ", sf_alpha)
	print("Estimated empirical sf exponent is ", lamb_hat)

	plt.xscale("log")
	plt.yscale("log")

	plt.ylabel("SF")
	plt.xlabel("Rank")

	plt.legend()
	plt.show()


def plot_sf_of_known_exponent():

	np.random.seed(3)
	alpha = 1.7
	ns = get_ranked_empirical_counts_from_infinite_power_law(alpha, N=1000)
	print(ns)
	ranks, sfs = get_survival_function(ns)
	print(sfs)
	# Plot the empirical sf
	plt.scatter(np.array(ranks), sfs, label="Empirical SF")

	sf_alpha = -1*(alpha-1)

	# Plot the predicted sf, based on alpha
	c = 0.1
	sf_hats = []
	for r in ranks:
		f = np.exp(c)*r**sf_alpha
		sf_hats.append(f)
	plt.plot(ranks, sf_hats, label="Expected SF")

	# Plot the estimated sf
	c, lamb_hat = ols_regression_cdf(ns, min_frequency=3)
	sf_hats = []
	for r in ranks:
		f = np.exp(c)*r**lamb_hat
		sf_hats.append(f)
	plt.plot(ranks, sf_hats, label="Estimated")	
	

	print("Expected empiricial sf exponent is ", sf_alpha)
	print("Estimated empirical sf exponent is ", lamb_hat)

	plt.xscale("log")
	plt.yscale("log")

	plt.ylabel("SF")
	plt.xlabel("Rank")

	plt.legend()
	plt.show()



plot_sf_of_known_exponent()