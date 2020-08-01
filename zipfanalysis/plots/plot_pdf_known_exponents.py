

import matplotlib.pyplot as plt
import numpy as np

from zipfanalysis.utilities.data_generators import get_counts_with_known_ranks, get_ranked_empirical_counts_from_infinite_power_law
from zipfanalysis.estimators.ols_regression_pdf import ols_regression_pdf


def plot_pdf_of_known_exponent_small_W():

	ns = get_counts_with_known_ranks(1.1, N=10000, W=15)
	ranks = range(1, len(ns)+1)

	plt.scatter(ranks, ns, label="Empirical Data")

	plt.xscale("log")
	plt.yscale("log")

	plt.ylabel("Frequency")
	plt.xlabel("Rank")

	c, lamb_hat = ols_regression_pdf(ns, min_frequency=1)

	print(lamb_hat)
	# Generate plot from estimated parameters
	f_hats = []
	for r in ranks:
		f = np.exp(c)*r**lamb_hat
		f_hats.append(f)
	plt.plot(ranks, f_hats, color="blue", label="PDF Fit")	
	plt.legend()
	plt.show()


def plot_pdf_of_known_exponent_large_W_known_ranks():

	ns = get_counts_with_known_ranks(1.1, N=10000, W=1000)
	ranks = range(1, len(ns)+1)

	plt.scatter(ranks, ns, label="Empirical Data")

	plt.xscale("log")
	plt.yscale("log")

	plt.ylabel("Frequency")
	plt.xlabel("Rank")

	c, lamb_hat = ols_regression_pdf(ns, min_frequency=1)

	print(lamb_hat)
	# Generate plot from estimated parameters
	f_hats = []
	for r in ranks:
		f = np.exp(c)*r**lamb_hat
		f_hats.append(f)
	plt.plot(ranks, f_hats, color="blue", label="PDF Fit")	
	plt.legend()
	plt.show()


def plot_pdf_of_known_exponent_empirical_ranks():

	ns = get_ranked_empirical_counts_from_infinite_power_law(1.3, N=10000)
	ranks = range(1, len(ns)+1)

	plt.scatter(ranks, ns, label="Empirical Data")

	plt.xscale("log")
	plt.yscale("log")

	plt.ylabel("Frequency")
	plt.xlabel("Rank")

	# Min frequcny is 1
	f_min = 1
	c, lamb_hat = ols_regression_pdf(ns, min_frequency=f_min)
	print("Estimated exponent is ", -1*lamb_hat)
	# Generate plot from estimated parameters
	f_hats = []
	for r in ranks:
		f = np.exp(c)*r**lamb_hat
		f_hats.append(f)
	plt.plot(ranks, f_hats, label="$PDF OLS fmin={}$".format(f_min))
	plt.legend()


	# Min frequency = 3
	f_min = 3
	
	c, lamb_hat = ols_regression_pdf(ns, min_frequency=f_min)
	print("Estimated exponent is ", -1*lamb_hat)
	# Generate plot from estimated parameters
	f_hats = []
	for r in ranks:
		f = np.exp(c)*r**lamb_hat
		f_hats.append(f)
	plt.plot(ranks, f_hats, label="$PDF OLS fmin={}$".format(f_min))	


	plt.show()



plot_pdf_of_known_exponent_empirical_ranks()