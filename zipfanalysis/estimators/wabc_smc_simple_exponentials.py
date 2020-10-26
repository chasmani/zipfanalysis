
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns

#1 Sample from an exponential
def get_exponential_data(lamb, size=None):
	return np.random.exponential(scale=1/lamb, size=size)

def get_mle(d):

	mean = sum(d)/len(d)
	mle = 1/mean
	return mle

def get_likelihood(d, lamb):
	return lamb**(len(d)) * np.exp(-1*lamb*sum(d))

def get_log_likelihood(d, lamb):
	return len(d)*np.log(lamb) - lamb*sum(d)

def plot_likelihood_function(d):

	lambs = np.linspace(0.4,1, 100)
	ls = []

	for lamb in lambs:
		l = get_likelihood(d, lamb)
		ls.append(l)

	ls = np.array(ls)
	ls = ls/sum(ls)*160

	mle = get_mle(d)
	print("mle is {}".format(mle))
	plt.axvline(mle)

	plt.xlabel("$\lambda$")
	plt.ylabel("$p(x|\lambda)$")

	plt.plot(lambs, ls, label="likelihood")
	#plt.show()




def wabc_smc_gaussian_kernel_exponential(x):

	n_particles = 1024
	survival_fraction = 0.1
	min_lamb=0.1
	max_lamb=5
	n_data = len(x)

	# 1A. Get particles by sampling from prior
	# Prior is uniform across minimum and maxmimum
	theta_0s = np.random.uniform(low=min_lamb, high=max_lamb, size=n_particles)

	ds = []
	# 1B Get distances by generating data from particles
	for k in range(n_particles):
		theta_k = theta_0s[k]
		z_k = get_exponential_data(theta_k, n_data)
		d_k = scipy.stats.wasserstein_distance(x, z_k)
		ds.append(d_k)

	print(theta_0s)
	print(ds)

	#1C, 2A and 2B Select tolerance and successful particles
	a_k, d_k, epsilon_k = extract_successful_trials(theta_0s, ds, survival_fraction*n_particles)

	for run_count in range(10):
		print("Running ", run_count)

		# 2C 
		# Resample from a
		particle_seeds = np.random.choice(a_k, size=n_particles, replace=True)
		current_sd = np.std(a_k/2)
		# Kernel - add some Gaussian noise
		noise = np.random.normal(loc=0, scale=current_sd, size=n_particles)
		thetas = particle_seeds + noise

		# Calculate the distances
		ds = []
		for k in range(n_particles):
			theta_k = thetas[k]
			z_k = get_exponential_data(theta_k, n_data)
			d_k = scipy.stats.wasserstein_distance(x, z_k)
			ds.append(d_k)
		
		a_k, d_k, epsilon_k = extract_successful_trials(thetas, ds, survival_fraction*n_particles)

	plt.hist(a_k)



	print("Successful particless: ", a_k)
	print("Current tolerance: ", epsilon_k)
	print(d_k)




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


def wabc_smc_exponential_lee_kernel(x):

	n_particles = 256
	survival_fraction = 0.2
	min_lamb=0.1
	max_lamb=5
	n_data = len(x)

	# 1A. Get particles by sampling from prior
	# Prior is uniform across minimum and maxmimum
	theta_0s = np.random.uniform(low=min_lamb, high=max_lamb, size=n_particles)

	ds = []
	# 1B Get distances by generating data from particles
	for k in range(n_particles):
		theta_k = theta_0s[k]
		z_k = get_exponential_data(theta_k, n_data)
		d_k = scipy.stats.wasserstein_distance(x, z_k)
		ds.append(d_k)

	#1C, 2A and 2B Select tolerance and successful particles
	a_k, d_k, epsilon_k = extract_successful_trials(theta_0s, ds, survival_fraction*n_particles)


	for run_count in range(15):
		print("Running ", run_count)
		epsilon_k_last = epsilon_k

		# SD of the data - use for the proposal distribution
		sd_k = np.std(a_k)
		# Posterior function
		pi_k = scipy.stats.gaussian_kde(a_k)
		# 2C 
		# Resample from a


		particle_seeds = np.random.choice(a_k, size=n_particles, replace=True)
		current_sd = np.std(a_k/2)
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
			theta, d = rejuvenate(theta_0, d_0, sd_k, x, epsilon_k, pi_k)
			thetas.append(theta)
			ds.append(d)


		a_k, d_k, epsilon_k = extract_successful_trials(thetas, ds, survival_fraction*n_particles)

		if abs(epsilon_k - epsilon_k_last) < 0.001:
			break

	sns.kdeplot(a_k, label="WABC")

	plt.xlim(0.4, 0.8)


	print("Successful particless: ", a_k)
	print("Current tolerance: ", epsilon_k)
	print(d_k)

def rejuvenate(theta_0, d_0, sd_k, x, epsilon_k, pi_k):

	
	# Step 1
	k_1 = 0
	r = 0
	while r < 2:
		
		theta_1 = np.random.normal(loc=theta_0, scale=sd_k)
		# Hacky - only continue if theta_1>0 - otehrwise it breaks. 
		# SHould cahnge the proposal distribution
		if theta_1 > 0:
			k_1 += 1
			z_1 = get_exponential_data(theta_1, size=len(x))
			d_1 = scipy.stats.wasserstein_distance(x, z_1)
			if d_1 <= epsilon_k:
				r += 1

	# Step 2
	k_2 = 0
	r = 0
	while r < 1:
		
		theta_2 = np.random.normal(loc=theta_1, scale=sd_k)
		# Hacky - only continue if theta_2>0 - otehrwise it breaks. 
		# SHould cahnge the proposal distribution
		if theta_2>0:
			k_2 += 1
			z_2 = get_exponential_data(theta_2, size=len(x))
			d_2 = scipy.stats.wasserstein_distance(z_2, x)
			if d_2 <= epsilon_k:
				r += 1

	hastings_ratio = pi_k.evaluate(theta_1)/pi_k.evaluate(theta_0) * k_2/(k_1+1)
	u = np.random.uniform()
	if u > hastings_ratio:
		return theta_1, d_1
	else:
		return theta_0, d_0






def basic_experiment():

	np.random.seed(8)

	d = get_exponential_data(lamb=0.6, size=200)
	wabc_smc_exponential_lee_kernel(d)
	plot_likelihood_function(d)
	
	plt.title("Posterior WABC with Data from an Exponential Model")
	plt.legend()
	plt.savefig("../plots/images/wabc_exponential.png")

	plt.show()




if __name__=="__main__":
	basic_experiment()