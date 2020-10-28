
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns

from zipfanalysis.utilities.data_generators import get_ranked_empirical_counts_from_infinite_power_law


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
	return successful_parameters, successful_distances, tolerance

def wabc_smc_zipf(ns, min_exponent=1.01, max_exponent=2.4, n_particles = 128, survival_fraction = 0.2):

	# 1A. Get particles by sampling from prior
	# Prior is uniform across minimum and maxmimum
	theta_0s = np.random.uniform(low=min_exponent, high=max_exponent, size=n_particles)
	n_data = len(ns)

	ds = []
	# 1B Get distances by generating data from particles
	for k in range(n_particles):
		theta_k = theta_0s[k]
		z_k = get_ranked_empirical_counts_from_infinite_power_law(theta_k, n_data)
		d_k = scipy.stats.wasserstein_distance(ns, z_k)
		ds.append(d_k)

	#1C, 2A and 2B Select tolerance and successful particles
	a_k, d_k, epsilon_k = extract_successful_trials(theta_0s, ds, survival_fraction*n_particles)

	d_sum = np.sum(d_k)

	for run_count in range(15):
		print("Running ", run_count)
		epsilon_k_last = epsilon_k
		d_sum = np.sum(d_k)
		print("D sum is ", d_sum)

		# SD of the data - use for the proposal distribution
		sd_k = np.std(a_k)
		# Posterior function
		pi_k = scipy.stats.gaussian_kde(a_k)
		# 2C 
		# Resample from a


		particle_seeds = np.random.choice(a_k, size=n_particles, replace=True)
		current_sd = np.std(a_k)
		# Kernel - add some Gaussian noise
		noise = np.random.normal(loc=0, scale=current_sd, size=n_particles)
		thetas = particle_seeds + noise


		# Randomly choose ancestors
		thetas = []
		ds = []

		ks = np.random.randint(len(a_k), size=n_particles)
		for k in ks:
			theta_0 = a_k[k]
			d_0 = a_k[k]
			theta, d = rejuvenate(theta_0, d_0, sd_k, ns, epsilon_k, pi_k)
			thetas.append(theta)
			ds.append(d)

		a_k, d_k, epsilon_k = extract_successful_trials(thetas, ds, survival_fraction*n_particles)


	sns.kdeplot(a_k, label="WABC")

def rejuvenate(theta_0, d_0, sd_k, x, epsilon_k, pi_k):

	
	# Step 1
	k_1 = 0
	r = 0
	while r < 2:
		
		theta_1 = np.random.normal(loc=theta_0, scale=sd_k)
		# Hacky - only continue if theta_1>1 - otehrwise it breaks. 
		# SHould cahnge the proposal distribution
		if theta_1 > 1.01:
			k_1 += 1
			z_1 = get_ranked_empirical_counts_from_infinite_power_law(theta_1, len(x))
			d_1 = scipy.stats.wasserstein_distance(x, z_1)
			if d_1 <= epsilon_k:
				r += 1

	# Step 2
	k_2 = 0
	r = 0
	while r < 1:
		
		theta_2 = np.random.normal(loc=theta_1, scale=sd_k)
		# Hacky - only continue if theta_2>1 - otehrwise it breaks. 
		# SHould cahnge the proposal distribution
		if theta_2>1.01:
			k_2 += 1
			z_2 = get_ranked_empirical_counts_from_infinite_power_law(theta_2, len(x))
			d_2 = scipy.stats.wasserstein_distance(z_2, x)
			if d_2 <= epsilon_k:
				r += 1

	hastings_ratio = pi_k.evaluate(theta_1)/pi_k.evaluate(theta_0) * k_2/(k_1+1)
	u = np.random.uniform()
	if u < hastings_ratio:
		return theta_1, d_1
	else:
		return theta_0, d_0


def rejuvenate_simple(theta_0, d_0, sd_k, x, epsilon_k, pi_k):

	
	# Step 1
	theta_1 = 0
	while theta_1 < 1.01:
		theta_1 = np.random.normal(loc=theta_0, scale=sd_k)

	hastings_ratio = min(1, pi_k.evaluate(theta_1)/pi_k.evaluate(theta_0))
	
	u = np.random.uniform()
	if u < hastings_ratio:
		z_1 = get_ranked_empirical_counts_from_infinite_power_law(theta_1, len(x))
		d_1 = scipy.stats.wasserstein_distance(x, z_1)
		if d_1 <= epsilon_k:
			return theta_1, d_1
	
	return theta_0, d_0


def wabc_smc_zipf_simple_kernel(ns, min_exponent=1.01, max_exponent=2, n_particles = 1024, survival_fraction = 0.4):

	# 1A. Get particles by sampling from prior
	# Prior is uniform across minimum and maxmimum
	theta_0s = np.random.uniform(low=min_exponent, high=max_exponent, size=n_particles)
	n_data = len(ns)

	ds = []
	# 1B Get distances by generating data from particles
	for k in range(n_particles):
		theta_k = theta_0s[k]
		z_k = get_ranked_empirical_counts_from_infinite_power_law(theta_k, n_data)
		d_k = scipy.stats.wasserstein_distance(ns, z_k)
		ds.append(d_k)

	#1C, 2A and 2B Select tolerance and successful particles
	a_k, d_k, epsilon_k = extract_successful_trials(theta_0s, ds, survival_fraction*n_particles)

	d_sum = np.sum(d_k)

	for run_count in range(5):
		print("Running ", run_count)
		epsilon_k_last = epsilon_k
		d_sum = np.sum(d_k)
		print("D sum is ", d_sum)

		# SD of the data - use for the proposal distribution
		sd_k = np.std(a_k)
		# Posterior function
		pi_k = scipy.stats.gaussian_kde(a_k)
		# 2C 
		# Resample from a


		particle_seeds = np.random.choice(a_k, size=n_particles, replace=True)
		current_sd = np.std(a_k)
		# Kernel - add some Gaussian noise
		noise = np.random.normal(loc=0, scale=current_sd, size=n_particles)
		thetas = particle_seeds + noise


		# Randomly choose ancestors
		thetas = []
		ds = []

		ks = np.random.randint(len(a_k), size=n_particles)
		for k in ks:
			theta_0 = a_k[k]
			d_0 = a_k[k]
			theta, d = rejuvenate(theta_0, d_0, sd_k, ns, epsilon_k, pi_k)
			thetas.append(theta)
			ds.append(d)

		a_k, d_k, epsilon_k = extract_successful_trials(thetas, ds, survival_fraction*n_particles)


	sns.kdeplot(a_k, label="WABC")


def basic_test():

	np.random.seed(5)

	alpha = 1.8
	ns = get_ranked_empirical_counts_from_infinite_power_law(alpha, N=10000)
	alpha_result = wabc_smc_zipf_simple_kernel(ns)
	plt.show()




if __name__=="__main__":
	basic_test()
