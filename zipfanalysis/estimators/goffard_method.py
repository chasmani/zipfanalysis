
import math
import time

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kde import KDEUnivariate
import scipy.stats
import pandas as pd
import seaborn as sns
import scipy.special
import scipy.spatial
import statsmodels.stats

from zipfanalysis.utilities.data_generators import get_ranked_empirical_counts_from_infinite_power_law

import csv

def append_to_csv(csv_list, output_filename):
	with open(output_filename, "a", newline='') as fp:
		a = csv.writer(fp, delimiter=';')
		data=[csv_list]
		a.writerows(data)

def get_normal_p(x_i, variance, mean):
	p_i = 1/(math.sqrt(2*math.pi*variance)) * np.exp(-0.5 * ((x_i - mean)/math.sqrt(variance))**2)
	return p_i

def plot_likelihood_normal(x, known_variance, prior_mean, prior_variance):

	posterior_mean = (prior_mean/prior_variance + sum(x)/known_variance) / (1/prior_variance + len(x)/known_variance)
	posterior_variance = 1 / (1/prior_variance + len(x)/known_variance)
	xs = np.linspace(2,6)
	ps = [get_normal_p(x, posterior_variance, posterior_mean) for x in xs]
	plt.plot(xs, ps, label="Actual Posterior")

def run_normal_test():

	prior_mean = 0
	prior_variance = 20
	known_variance = 10
	actual_mean = 4
	z = np.random.normal(loc=actual_mean, scale=math.sqrt(known_variance), size=100)
	plot_likelihood_normal(z, known_variance, prior_mean, prior_variance)


def goffard_normal(x, known_variance, prior_mean, prior_variance):

	n_particles = 1024
	known_sd = math.sqrt(known_variance)
	n_data = len(x)
	alpha=0.2

	# 1. Draw particles from prior
	a_k = np.random.normal(loc=prior_mean, scale=math.sqrt(prior_variance), size=n_particles)

	#2. KDE of particles
	weights = np.array([0.1]*len(a_k))
	kde = scipy.stats.gaussian_kde(a_k, weights=weights)
	
	epsilon = 10000

	for g in range(5):
		print("Generation ", g)
		d_ks = []
		a_ks = []
		for k in range(n_particles):
			hit = False
			while not hit:
				a_k = kde.resample(size=1)[0][0]
				z_k = np.random.normal(loc=a_k, scale=known_sd, size=n_data)
				d_k = scipy.stats.wasserstein_distance(z_k, x)
				if d_k <= epsilon:
					d_ks.append(d_k)
					a_ks.append(a_k)
					hit=True

		# set new epsilon
		successful_as,successful_ds, epsilon = extract_successful_trials(a_ks, d_ks, alpha)

		print("Epsilon is now ", epsilon)

		ws = get_weights(a_ks, d_ks, prior_mean, prior_variance, kde, epsilon)

		kde = scipy.stats.gaussian_kde(a_ks, weights=ws)




	xs = np.linspace(2,6)
	kdes = [kde.evaluate(x_i) for x_i in xs]
	plt.plot(xs, kdes, label="WABC")


def extract_successful_trials(parameters, distances, alpha):
	"""
	Get close trial results, at least as many as "required_accepted_parameters"
	"""
	# Choose a low tolerance to begin
	tolerance = 1
	finished = False

	required_accepted_parameters = alpha * len(parameters)
	# Keep on expanding the tolerance until we get enough successful trials
	while not finished:

		successes = np.absolute(distances) < tolerance
		success_count = np.count_nonzero(successes)
		if success_count<required_accepted_parameters:
			tolerance = tolerance*1.1
		else:
			finished = True

	successful_parameters = np.array(parameters)[successes]
	successful_distances = np.array(distances)[successes]
	return successful_parameters, successful_distances, tolerance


def get_weights(a_ks, d_ks, prior_mean, prior_variance, kde, epsilon):

	ws = []
	for k in range(len(a_ks)):
		if d_ks[k] > epsilon:
			ws.append(0)
		else:
			a_k = a_ks[k]
			prior_p = get_normal_p(a_k, prior_variance, prior_mean)
			kde_p = kde.evaluate(a_k)[0]
			w_k = prior_p/kde_p
			ws.append(w_k)
	return np.array(ws)

def get_weights_gamma_prior(a_ks, d_ks, prior_alpha, prior_beta, kde, epsilon):

	ws = []
	for k in range(len(a_ks)):
		if d_ks[k] > epsilon:
			ws.append(0)
		else:
			a_k = a_ks[k]
			prior_p = get_gamma_p(a_k, prior_alpha, prior_beta)
			kde_p = kde.evaluate(a_k)[0]
			w_k = prior_p/kde_p
			ws.append(w_k)
	return np.array(ws)	





def test_goffard():

	np.random.seed(1)
	prior_mean = 0
	prior_variance = 20
	known_variance = 10
	actual_mean = 4
	x = np.random.normal(loc=actual_mean, scale=math.sqrt(known_variance), size=100)
	goffard_normal(x, known_variance, prior_mean, prior_variance)
	plot_likelihood_normal(x, known_variance, prior_mean, prior_variance)	
	plt.legend()
	plt.xlabel(r"$\mu$")
	plt.ylabel(r"$p(\mu|d)$")
	plt.title("WABC Goffard Method Normal Model known Variance\n with Normal Conjugate Prior")
	plt.savefig("../plots/images/goffard_normal.png")
	plt.show()
	


def goffard_exponential(x, prior_alpha, prior_beta):

	n_particles = 1024
	n_data = len(x)
	alpha=0.2

	# 1. Draw particles from prior
	a_k = np.random.gamma(shape=prior_alpha, scale=1/prior_beta, size=n_particles)

	#2. KDE of particles
	weights = np.array([0.1]*len(a_k))
	kde = scipy.stats.gaussian_kde(a_k, weights=weights)
	
	epsilon = 10000

	for g in range(3):
		print("Generation ", g)
		d_ks = []
		a_ks = []
		for k in range(n_particles):
			hit = False
			while not hit:
				a_k = kde.resample(size=1)[0][0]
				# a_k below zero can be ignored I think- they won't hit
				if a_k > 0:
					z_k = np.random.exponential(scale=1/a_k, size=n_data)
					d_k = scipy.stats.wasserstein_distance(z_k, x)
					if d_k <= epsilon:
						d_ks.append(d_k)
						a_ks.append(a_k)
						hit=True

		# set new epsilon
		successful_as,successful_ds, epsilon = extract_successful_trials(a_ks, d_ks, alpha)

		print("Epsilon is now ", epsilon)

		ws = get_weights_gamma_prior(a_ks, d_ks, prior_alpha, prior_beta, kde, epsilon)
		print(ws)

		kde = scipy.stats.gaussian_kde(a_ks, weights=ws)




	xs = np.linspace(0,1)
	kdes = [kde.evaluate(x_i) for x_i in xs]
	plt.plot(xs, kdes, label="WABC")


def wasserstein_on_all_data_points(z, x):

	z_all = []
	for i in range(len(z)):
		z_all.append(i*z[i])

	x_all = []
	for i in range(len(x)):
		x_all.append(i*x[i])

	return scipy.stats.wasserstein_distance(z_all, x_all)



def plot_likelihood_exponential(x, prior_alpha, prior_beta):

	posterior_alpha = prior_alpha + len(x)
	posterior_beta = prior_beta + sum(x)

	xs = np.linspace(0,2)
	ps = [get_gamma_p(x, posterior_alpha, posterior_beta) for x in xs]
	plt.plot(xs, ps, label="Actual Posterior")

def get_gamma_p(x, alpha, beta):
	p = (beta**alpha) /scipy.special.gamma(alpha) * x**(alpha-1) * np.exp(-1*beta*x)
	return p

def test_goffard_exponential():

	np.random.seed(1)
	# Gamma is a conjugate prior
	prior_alpha = 1
	prior_beta = 1

	actual_lamb = 0.6
	x = np.random.exponential(scale=1/actual_lamb, size=100)
	plot_likelihood_exponential(x, prior_alpha, prior_beta)
	goffard_exponential(x, prior_alpha, prior_beta)
	plt.legend()

	plt.xlabel(r"$\lambda$")
	plt.ylabel(r"$p(\lambda|d)$")
	plt.title("WABC Goffard Method \nExponential Model with Gamma Conjugate Prior")
	plt.savefig("../plots/images/goffard_exponential.png")
	plt.show()


def goffard_zipf(x, min_exponent, max_exponent, n_particles, success_proportion, bandwidth, generations):


	n_data = sum(x)

	# 1. Draw particles from prior
	a_k = np.random.uniform(low=min_exponent, high=max_exponent, size=n_particles)

	#2. KDE of particles
	weights = np.array([0.1]*len(a_k))
	kde = scipy.stats.gaussian_kde(a_k, weights=weights)
	
	epsilon = 10000

	for g in range(generations):
		print("Generation ", g)

		d_ks = []
		a_ks = []
		for k in range(n_particles):
			hit = False
			while not hit:
				a_k = kde.resample(size=1)[0][0]
				# a_k below zero can be ignored I think- they won't hit
				if a_k > min_exponent and a_k < max_exponent:
					z_k = get_ranked_empirical_counts_from_infinite_power_law(a_k, N=n_data)
					d_k = scipy.stats.wasserstein_distance(z_k, x)
					if d_k <= epsilon:
						d_ks.append(d_k)
						a_ks.append(a_k)
						hit=True



		# set new epsilon
		successful_as, successful_ds, epsilon = extract_successful_trials(a_ks, d_ks, success_proportion)
		epsilon = max(successful_ds)
		ws = get_weights_uniform_prior(a_ks, d_ks, min_exponent, max_exponent, kde, epsilon)

		kde = scipy.stats.gaussian_kde(a_ks, weights=ws, bw_method=bandwidth)


		xs = np.linspace(1.01, 3, num=10000)
		kdes = []
		mle = 0
		max_k = 0
		for x_i in xs:
			value = kde.evaluate(x_i)
			kdes.append(value)
			if value>max_k:
				max_k = value
				mle = x_i
		
	
		if g%1==0:
			plt.plot(xs, kdes, label="WABC {}".format(g))

	plt.hist(a_ks, bins=50)
	plt.xlim(1.7, 1.9)
	plt.axvline(mle)
	plt.legend()
	

	return mle



def get_weights_uniform_prior(a_ks, d_ks, min_exponent, max_exponent, kde, epsilon):

	ws = []
	for k in range(len(a_ks)):
		if d_ks[k] > epsilon:
			ws.append(0)
		else:
			a_k = a_ks[k]
			
			if a_k < min_exponent or a_k > max_exponent:
				ws.append(0)
			else:
				prior_p = 1 # uniform
				kde_p = kde.evaluate(a_k)[0]
				w_k = prior_p/kde_p
				ws.append(w_k)
	return np.array(ws)	

def weighted_variance(a_ks, ws):

	print(np.cov(a_ks, aweights=ws))
	




def basic_test():

	np.random.seed(3)
	results = []
	bandwidth = 1
	success_proportion = 0.1
	generations = 4
	n_particles = 256
	alpha = 1.8
	ns = get_ranked_empirical_counts_from_infinite_power_law(alpha, N=10000)
	alpha_result = goffard_zipf(ns, 1.03, 3,
					n_particles, success_proportion, bandwidth, generations)
	plt.show()


def check_wasserstein_distance():

	alpha_tests = list(np.linspace(1.78,1.82, 20))*20
	d_ns = []
	d_0s = []

	for alpha_test in alpha_tests:
		np.random.seed()
		alpha = 1.8
		ns_1 = np.array(get_ranked_empirical_counts_from_infinite_power_law(alpha, N=10000))
		ns_2 = np.array(get_ranked_empirical_counts_from_infinite_power_law(alpha_test, N=10000))
		d_n = scipy.stats.wasserstein_distance(ns_1, ns_2)
		d_ns.append(d_n)
		d_0 = scipy.stats.wasserstein_distance(ns_1, ns_2)
		d_0, p = scipy.stats.ks_2samp(ns_1, ns_2)
		d_0, p = scipy.spatial.distance.jensenshannon(ns_1/sum(ns_1), ns_2/sum(ns_2))


		d_0s.append(d_0)
		#plt.scatter(alpha_tests, d_0s, label="all points")
	df = pd.DataFrame(
    	{'alpha': alpha_tests,
     	'wasserstein': d_0s,
    	})

	sns.boxplot(x="alpha", y="wasserstein", data=df)
	#plt.scatter(alpha_tests, d_ns, label="n points")
	plt.legend()
	plt.show()

def run_simulations():

	results_filename = "goffard_overnight_results_more_fields.csv"

	bandwidth = 1
	success_proportion = 0.1
	generations = 4
	n_particles = 512

	results = []
	for seed in range(8,100):
		for alpha in np.linspace(1.1, 2.3, 10):
		
			try:
				print(alpha, seed)
				np.random.seed(seed)
				ns = get_ranked_empirical_counts_from_infinite_power_law(alpha, N=10000)
				start=time.time()
				alpha_result = goffard_zipf(ns, 1.03, 3,
					n_particles, success_proportion, bandwidth, generations)
				end = time.time()
				csv_list = [seed, alpha, "goffard", bandwidth, success_proportion, generations, n_particles, alpha_result, end-start]
			except Exception as e:
				csv_list = [seed, alpha, "goffard", str(e)]

			append_to_csv(csv_list, results_filename)
			
# To do
# Play around with bandwidth, generations, n_particles, alphas, other distance measures
# Maybe just use a histoigram in the end to get the mle, rather than kde
# Try other methods/kernels, now you know how to make them work

if __name__=="__main__":
	run_simulations()