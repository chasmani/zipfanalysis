
import math

import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def get_normal_p(x_i, variance, mean):
	p_i = 1/(math.sqrt(2*math.pi*variance)) * np.exp(-0.5 * ((x_i - mean)/math.sqrt(variance))**2)
	return p_i

def plot_likelihood_normal(x, known_variance, prior_mean, prior_variance):

	posterior_mean = (prior_mean/prior_variance + sum(x)/known_variance) / (1/prior_variance + len(x)/known_variance)
	posterior_variance = 1 / (1/prior_variance + len(x)/known_variance)
	xs = np.linspace(2,6)
	ps = [get_normal_p(x, posterior_variance, posterior_mean) for x in xs]
	plt.plot(xs, ps, label="Actual Posterior")

def my_wabc_normal(x, known_variance, prior_mean, prior_variance):

	generations = 4
	n_particles = 1024
	survival_fraction = 0.2
	n_data = len(x)

	prior_sd = math.sqrt(prior_variance)
	known_sd = math.sqrt(known_variance)

	a_ks = np.random.normal(loc=prior_mean, scale=prior_sd, size=n_particles)
	ds = []
	# Get distances by generating data from particles
	for k in range(n_particles):
		a_k = a_ks[k]
		z_k = np.random.normal(loc=a_k, scale=known_sd, size=n_data)
		d_k = scipy.stats.wasserstein_distance(x, z_k)
		ds.append(d_k)

	for g in range(generations):
		print("Generation ", g)


		# Select a new tolerance based on alpha* N particles being selected		
		tolerance = get_new_tolerance(ds, survival_fraction)

		# kde exact form shouldn't matter, only that theta>thet
		kde = scipy.stats.gaussian_kde(a_ks, bw_method=1)

		a_ks, ds = rejuvenate(x, a_ks, kde, tolerance, n_particles, n_data, prior_mean, prior_variance, known_variance)
	plt.hist(a_ks, density=True)
	sns.kdeplot(a_ks)

# TO FIX - ACCEPT A FIXED PERCENTAGE EACH TIME - ONLY GENERATE ONE POINT FROM EACH PARTICLE
# EITHER SWITCH OR STAY THE SAME. BUT THEN ONLY ACCEPT A PROPORTION OF SAMPLES

def my_wabc_normal(x, known_variance, prior_mean, prior_variance):

	generations = 4
	n_particles = 1024
	survival_fraction = 0.2
	n_data = len(x)

	prior_sd = math.sqrt(prior_variance)
	known_sd = math.sqrt(known_variance)

	a_ks = np.random.normal(loc=prior_mean, scale=prior_sd, size=n_particles)
	ds = []
	# Get distances by generating data from particles
	for k in range(n_particles):
		a_k = a_ks[k]
		z_k = np.random.normal(loc=a_k, scale=known_sd, size=n_data)
		d_k = scipy.stats.wasserstein_distance(x, z_k)
		ds.append(d_k)

	for g in range(generations):
		print("Generation ", g)


		# Select a new tolerance based on alpha* N particles being selected		
		tolerance = get_new_tolerance(ds, survival_fraction)

		# kde exact form shouldn't matter, only that theta>thet
		kde = scipy.stats.gaussian_kde(a_ks, bw_method=1)

		a_ks, ds = rejuvenate(x, a_ks, kde, tolerance, n_particles, n_data, prior_mean, prior_variance, known_variance)
	plt.hist(a_ks, density=True)
	sns.kdeplot(a_ks)

def my_wabc_normal_2(x, known_variance, prior_mean, prior_variance):

	generations = 5
	n_particles = 1024
	survival_fraction = 0.2
	n_data = len(x)

	prior_sd = math.sqrt(prior_variance)
	known_sd = math.sqrt(known_variance)

	a_ks = np.random.normal(loc=prior_mean, scale=prior_sd, size=n_particles)
	ds = []
	# Get distances by generating data from particles
	for k in range(n_particles):
		a_k = a_ks[k]
		z_k = np.random.normal(loc=a_k, scale=known_sd, size=n_data)
		d_k = scipy.stats.wasserstein_distance(x, z_k)
		ds.append(d_k)

	for g in range(generations):
		print("Generation ", g)

		# kde exact form shouldn't matter, only that theta>thet
		kde = scipy.stats.gaussian_kde(a_ks, bw_method=0.5)

		# Rejuvenate
		a_ks = rejuvenate_2(x, a_ks, kde, n_particles, n_data, prior_mean, prior_variance, known_variance)

		ds = []
		for k in range(n_particles):
			a_k = a_ks[k]
			z_k = np.random.normal(loc=a_k, scale=known_sd, size=n_data)
			d_k = scipy.stats.wasserstein_distance(x, z_k)
			ds.append(d_k)		

		# Get proportion of successful params
		a_ks, ds, new_tolerance = get_successful_params(a_ks, ds, survival_fraction)
	
	plt.hist(a_ks, density=True)
	sns.kdeplot(a_ks)

def my_wabc_normal_3(x, known_variance, prior_mean, prior_variance):

	generations = 5
	n_particles = 1024
	survival_fraction = 0.2
	n_data = len(x)

	prior_sd = math.sqrt(prior_variance)
	known_sd = math.sqrt(known_variance)

	a_ks = np.random.normal(loc=prior_mean, scale=prior_sd, size=n_particles)
	ds = []
	# Get distances by generating data from particles
	for k in range(n_particles):
		a_k = a_ks[k]
		z_k = np.random.normal(loc=a_k, scale=known_sd, size=n_data)
		d_k = scipy.stats.wasserstein_distance(x, z_k)
		ds.append(d_k)

	for g in range(generations):
		print("Generation ", g)

		# kde exact form shouldn't matter, only that theta>thet
		kde = scipy.stats.gaussian_kde(a_ks, bw_method=1)

		tolerance = get_new_tolerance(ds, survival_fraction)

		a_ks, ds = rejuvenate_3(x, a_ks, ds, tolerance, kde, n_particles, n_data, prior_mean, prior_variance, known_variance)
		# Rejuvenate
	
	plt.hist(a_ks, density=True)
	sns.kdeplot(a_ks)

def rejuvenate(x, a_ks, kde, tolerance, n_particles, n_data, prior_mean, prior_variance, known_variance):

	successes = 0
	a_ks_next = []
	ds_next = []
	std_k = np.std(a_ks)
	prior_sd = math.sqrt(prior_variance)
	known_sd = math.sqrt(known_variance)
	while successes < n_particles:
		theta = np.random.choice(a_ks)
		theta_prime = np.random.normal(loc=theta, scale=std_k)
		z_prime = np.random.normal(loc=theta_prime, scale=known_sd, size=n_data)
		d_prime = scipy.stats.wasserstein_distance(x, z_prime)
		if d_prime <= tolerance:
			prior_theta = scipy.stats.norm.pdf(theta, loc=prior_mean, scale=prior_sd)
			prior_theta_prime = scipy.stats.norm.pdf(theta_prime, loc=prior_mean, scale=prior_sd)
			kde_theta = kde.evaluate(theta)
			kde_theta_prime = kde.evaluate(theta_prime)
			h = min(1, prior_theta/prior_theta_prime * kde_theta_prime/kde_theta)
			u = np.random.uniform()
			if u < h:
				successes += 1
				a_ks_next.append(theta_prime)
				ds_next.append(d_prime)

	return a_ks_next, ds_next

def rejuvenate_2(x, a_ks, kde, n_particles, n_data, prior_mean, prior_variance, known_variance):

	# n_particles times
	# Draw from particles randomly
	a_ks_next = []
	std_k = np.std(a_ks)
	prior_sd = math.sqrt(prior_variance)
	known_sd = math.sqrt(known_variance)
	
	a_ks_next = []
	for k in range(n_particles):
		theta = np.random.choice(a_ks)
		theta_prime = theta_prime = np.random.normal(loc=theta, scale=2*std_k)
		kde_theta = kde.evaluate(theta)
		kde_theta_prime = kde.evaluate(theta_prime)	
		prior_theta = scipy.stats.norm.pdf(theta, loc=prior_mean, scale=prior_sd)
		prior_theta_prime = scipy.stats.norm.pdf(theta_prime, loc=prior_mean, scale=prior_sd)
		h = min(1, prior_theta/prior_theta_prime * kde_theta_prime/kde_theta)
		u = np.random.uniform()
		if u < h:
			a_ks_next.append(theta_prime)
		else:
			a_ks_next.append(theta)

	return a_ks_next

def rejuvenate_3(x, a_ks, ds, tolerance, kde, n_particles, n_data, prior_mean, prior_variance, known_variance):

	# n_particles times
	# Draw from particles randomly
	a_ks_next = []
	ds_next = []
	std_k = np.std(a_ks)
	prior_sd = math.sqrt(prior_variance)
	known_sd = math.sqrt(known_variance)
	

	for k in range(n_particles):
		theta = np.random.choice(a_ks)
		theta_prime = theta_prime = np.random.normal(loc=theta, scale=std_k)
		kde_theta = kde.evaluate(theta)
		kde_theta_prime = kde.evaluate(theta_prime)	
		prior_theta = scipy.stats.norm.pdf(theta, loc=prior_mean, scale=prior_sd)
		prior_theta_prime = scipy.stats.norm.pdf(theta_prime, loc=prior_mean, scale=prior_sd)
		h = min(1, prior_theta/prior_theta_prime * kde_theta_prime/kde_theta)
		u = np.random.uniform()
		new_theta = theta
		new_d = ds[k]
		if u < h:
			z_prime = np.random.normal(loc=theta_prime, scale=known_sd, size=n_data)
			d_prime = scipy.stats.wasserstein_distance(x, z_prime)
			if d_prime < tolerance:
				new_theta = theta_prime
				new_d = d_prime
		a_ks_next.append(new_theta)
		ds_next.append(new_d)

	return a_ks_next, ds_next



def get_new_tolerance(distances, survival_fraction):

	sorted_distances = sorted(distances)
	# Round up the number of acceptances
	accepted = math.ceil(survival_fraction*len(distances))
	new_tolerance = sorted_distances[accepted-1]
	return new_tolerance

def get_successful_params(a_ks, distances, survival_fraction):

	new_aks = []
	new_ds = []

	sorted_distances = sorted(distances)
	# Round up the number of acceptances
	accepted = math.ceil(survival_fraction*len(distances))
	new_tolerance = sorted_distances[accepted-1]
	for j in range(len(distances)):
		d = distances[j]
		if d <= new_tolerance:
			new_aks.append(a_ks[j])
			new_ds.append(distances[j])

	print(new_aks)
	return new_aks, new_ds, new_tolerance


def basic_test_normal():

	np.random.seed(1)
	prior_mean = 0
	prior_variance = 20
	known_variance = 10
	actual_mean = 4
	x = np.random.normal(loc=actual_mean, scale=math.sqrt(known_variance), size=100)
	my_wabc_normal_2(x, known_variance, prior_mean, prior_variance)
	plot_likelihood_normal(x, known_variance, prior_mean, prior_variance)	
	plt.show()


basic_test_normal()

