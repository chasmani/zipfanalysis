
import math
import time
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.special

import csv


def convert_observations_into_ranked_empirical_counts(samples):

	counts = Counter(samples)
	n = [v for k,v in counts.most_common()]
	return n

######################
# Infinite event set

def generate_samples_from_infinite_power_law(exponent, N):

	xs_np = np.random.zipf(a=exponent, size = N)
	return xs_np


def get_ranked_empirical_counts_from_infinite_power_law(exponent, N):

	xs = generate_samples_from_infinite_power_law(exponent, N)
	n = convert_observations_into_ranked_empirical_counts(xs)
	return n


def append_to_csv(csv_list, output_filename):
	with open(output_filename, "a", newline='') as fp:
		a = csv.writer(fp, delimiter=';')
		data=[csv_list]
		a.writerows(data)


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

	lambs = np.linspace(0.4,0.8, 100)
	ls = []

	for lamb in lambs:
		l = get_likelihood(d, lamb)
		ls.append(l)

	ls = np.array(ls)

	bar_width = 0.4/100

	area = 0
	for i in range(100):
		area += lambs[i] * ls[i]
	# Noralise it
	ls = ls/(area*bar_width)


	mle = get_mle(d)
	print("mle is {}".format(mle))
	plt.axvline(mle)

	plt.xlabel("$\lambda$")
	plt.ylabel("$p(x|\lambda)$")

	plt.plot(lambs, ls, label="likelihood")
	#plt.show()

def plot_posterior_exponential(x, prior_alpha, prior_beta):

	posterior_alpha = prior_alpha + len(x)
	posterior_beta = prior_beta + sum(x)

	print(posterior_alpha)
	xs = np.linspace(0.5,0.7, 200)
	ps = [get_gamma_p(x, posterior_alpha, posterior_beta) for x in xs]

	plt.plot(xs, ps, label="Actual Posterior")

def get_gamma_p(x, alpha, beta):

	p = (beta**alpha) /scipy.special.gamma(alpha) * x**(alpha-1) * np.exp(-1*beta*x)
	return p

def beaumont_pmc_gamma_prior(x, prior_alpha, prior_beta):

	n_particles = 256
	survival_fraction = 0.6
	min_lamb=0.1
	max_lamb=2
	n_data = len(x)
	generations = 10

	# 1. Generate thetas from prior
	ds = []
	thetas = np.random.gamma(shape=prior_alpha, scale=1/prior_beta, size=n_particles)
	for j in range(n_particles):
		theta_j = thetas[j]
		z_j = get_exponential_data(theta_j, size=n_data)
		d_j = scipy.stats.wasserstein_distance(x, z_j)
		ds.append(d_j)

	ws = np.full(n_particles, 1/n_particles)

	var = 2*np.var(thetas)
	sd = np.sqrt(var)
	#2. 
	print(var)

	ws_sum = 1
	for g in range(generations):
		tolerance = get_new_tolerance(ds, survival_fraction)
		print("Tolerance is ", tolerance)
		ps = ws/ws_sum
		ds_next = []
		thetas_next = []
		ws_next = []
		test_count = 0
		hit_count = 0
		for i in range(n_particles):
			hit=False
			while not hit:
				test_count += 1
				print(hit_count, test_count)
				j = np.random.choice(n_particles, p=ps)
				theta_j = thetas[j]
				theta_prime = np.random.normal(loc=theta_j, scale=sd)
				if theta_prime > 0 and theta_prime<2:
					z_j = get_exponential_data(theta_prime, size=n_data)
					d_j = scipy.stats.wasserstein_distance(x, z_j)
					if d_j <= tolerance:
						thetas_next.append(theta_prime)
						ds_next.append(d_j)
						hit=True
						hit_count += 1
						prior_i = get_gamma_p(theta_prime, prior_alpha, prior_beta)						
						summation = beaumont_sum(ws, sd, theta_prime, thetas)
						w_i = prior_i/summation
						ws_next.append(w_i)						

		thetas = thetas_next
		ds = ds_next

		#plt.scatter(thetas, ds)
		#plt.show()

		ws = ws_next
		var = 2*np.cov(thetas, aweights=ws)
		sd= math.sqrt(var)
		ws_sum = np.sum(ws)
		print(g, tolerance, sd, test_count)
		"""
		plot_posterior_exponential(x, prior_alpha, prior_beta)
	
		# PLot kde
		kde = scipy.stats.gaussian_kde(thetas, weights=ws)
		xs = np.linspace(0.5,0.7)
		kdes = [kde.evaluate(x_i) for x_i in xs]
		plt.plot(xs, kdes, label="WABC")
		
		#sns.kdeplot(thetas, label=g)

		plt.show()
		"""

	plot_posterior_exponential(x, prior_alpha, prior_beta)
	
		# PLot kde
	kde = scipy.stats.gaussian_kde(thetas, weights=ws)
	xs = np.linspace(0.5,0.7)
	kdes = [kde.evaluate(x_i) for x_i in xs]
	plt.plot(xs, kdes, label="WABC")


# POTENTIAL IMPROVEMENT
# SET TOLERANCE DYNAMICALLY - GENERATE DATA POINTS, ACCEPT 10%

def beaumont_sum(ws_j, sd, theta_i, thetas_j):

	summation = 0
	for j in range(len(ws_j)):
		stand_norm_var = (theta_i - thetas_j[j]) / sd
		#summation += ws_j[j] * scipy.stats.norm.pdf(stand_norm_var)	
		summation += ws_j[j] * scipy.stats.t.pdf(theta_i, df=99, loc=thetas_j[j], scale=sd)	
			
	return summation


def get_new_tolerance(distances, survival_fraction):

	sorted_distances = sorted(distances)
	# Round up the number of acceptances
	accepted = math.ceil(survival_fraction*len(distances))
	new_tolerance = sorted_distances[accepted-1]
	return new_tolerance


def beaumont_pmc_gamma_prior_dynamic_tolerance(x, prior_alpha, prior_beta):

	survival_fraction = 0.05
	n_particles = 250
	n_seeds = int(n_particles/survival_fraction)
	min_theta=0.001
	max_theta=2
	n_data = len(x)
	generations = 10

	# 1. Generate thetas from prior
	ds = []
	thetas = np.random.gamma(shape=prior_alpha, scale=1/prior_beta, size=n_particles)
	for j in range(n_particles):
		theta_j = thetas[j]
		z_j = get_exponential_data(theta_j, size=n_data)
		d_j = scipy.stats.wasserstein_distance(x, z_j)
		ds.append(d_j)

	ws = np.full(n_particles, 1/n_particles)

	var = 2*np.var(thetas)
	sd = np.sqrt(var)
	#2. 
	print(var)

	for g in range(generations):
		ps = ws/np.sum(ws)
		check_count = 0
		
		theta_stars = []
		d_stars = []

		# Generate theta_stars / seeds 
		for i in range(n_seeds):
			hit = False
			check_count += 1
			print(check_count)
			# "Hit" is just valid parameters
			while not hit:

				j = np.random.choice(len(thetas), p=ps)
				theta_j = thetas[j]
				theta_star = np.random.normal(loc=theta_j, scale=sd)

				if theta_star > min_theta and theta_star<max_theta:
					z_star = get_exponential_data(theta_star, size=n_data)
					d_star = scipy.stats.wasserstein_distance(x, z_star)
					theta_stars.append(theta_star)
					d_stars.append(d_star)
					hit = True

		# Get the next thetas, next ds and new tolerance
		# Pick the n_particles theta_stars with the lowest distances
		thetas_next, ds_next, tolerance = get_successful_stars(d_stars, theta_stars, n_particles)

		# Get the new weights
		ws_next = get_new_weights_gamma_prior(thetas_next, thetas, ws, prior_alpha, prior_beta, sd)

		thetas = thetas_next
		ws = ws_next

		var = 2*np.cov(thetas, aweights=ws)
		sd= math.sqrt(var)
		
		# PLot kde
		kde = scipy.stats.gaussian_kde(thetas, weights=ws)
		xs = np.linspace(0.5,0.7)
		kdes = [kde.evaluate(x_i) for x_i in xs]
	plt.plot(xs, kdes, label="WABC")

	plot_posterior_exponential(x, prior_alpha, prior_beta)
	
def get_successful_stars(d_stars, theta_stars, n_particles):

	print("Getting successes")
	sorted_distances = sorted(d_stars)
	# Round up the number of acceptances
	new_tolerance = sorted_distances[n_particles-1]

	thetas_survived = []
	ds_survived = []

	for j in range(len(d_stars)):
		if d_stars[j] <= new_tolerance:
			thetas_survived.append(theta_stars[j])
			ds_survived.append(d_stars[j])
	return thetas_survived, ds_survived, new_tolerance


def get_new_weights_gamma_prior(thetas_next, thetas, ws, prior_alpha, prior_beta, sd):

	print("Getting weights")
	ws_next = []
	for j in range(len(thetas_next)):
		theta_prime = thetas_next[j]
		prior_i = get_gamma_p(theta_prime, prior_alpha, prior_beta)						
		summation = beaumont_sum(ws, sd, theta_prime, thetas)
		w_i = prior_i/summation
		ws_next.append(w_i)		
	return ws_next

def get_new_weights_uniform_prior(thetas_next, thetas, ws, sd):

	print("Getting weights")
	ws_next = []
	for j in range(len(thetas_next)):
		theta_prime = thetas_next[j]
		prior_i = 1						
		summation = beaumont_sum(ws, sd, theta_prime, thetas)
		w_i = prior_i/summation
		ws_next.append(w_i)		
	return ws_next




def sanity_check():

	x = get_exponential_data(lamb=0.6, size=200)
	z_1 = get_exponential_data(lamb=0.62, size=200)
	z_2 = get_exponential_data(lamb=0.9, size=200)
	d_1 = scipy.stats.wasserstein_distance(x, z_1)
	d_2 = scipy.stats.wasserstein_distance(x, z_2)
	print(d_1, d_2)



def beaumont_pmc_zipf(ns, n_particles, survival_fraction, generations):

	
	min_lamb=1.001
	max_lamb=3
	n_data = sum(ns)


	# 1. Generate thetas from prior
	ds = []
	thetas = np.random.uniform(low=min_lamb, high=max_lamb, size=n_particles)
	for j in range(n_particles):
		theta_j = thetas[j]
		z_j = get_ranked_empirical_counts_from_infinite_power_law(theta_j, N=n_data)
		d_j = scipy.stats.wasserstein_distance(ns, z_j)
		ds.append(d_j)

	ws = np.full(n_particles, 1/n_particles)

	var = 2*np.var(thetas)
	sd = np.sqrt(var)


	ws_sum = 1
	for g in range(generations):
		print("Generation ", g)
		tolerance = get_new_tolerance(ds, survival_fraction)
		ps = ws/ws_sum
		ds_next = []
		thetas_next = []
		ws_next = []
		test_count = 0
		hit_count = 0
		for i in range(n_particles):
			hit=False
			while not hit:
				test_count += 1
				j = np.random.choice(n_particles, p=ps)
				theta_j = thetas[j]
				theta_prime = np.random.normal(loc=theta_j, scale=sd)
				if theta_prime > min_lamb and theta_prime<max_lamb:
					z_j = get_ranked_empirical_counts_from_infinite_power_law(theta_prime, N=n_data)
					d_j = scipy.stats.wasserstein_distance(ns, z_j)
					if d_j <= tolerance:
						thetas_next.append(theta_prime)
						ds_next.append(d_j)
						hit=True
						hit_count += 1
						prior_i = 1						
						summation = beaumont_sum(ws, sd, theta_prime, thetas)
						w_i = prior_i/summation
						ws_next.append(w_i)						

		thetas = thetas_next
		ds = ds_next

		ws = ws_next
		plt.scatter(thetas, ds)
		plt.show()
		var = 2*np.cov(thetas, aweights=ws)
		sd= math.sqrt(var)
		ws_sum = np.sum(ws)

		# PLot mle = xs[np.argmax(kdes)]
	
	kde = scipy.stats.gaussian_kde(thetas, weights=ws)
	xs = np.linspace(1.001,2, 10000)
	kdes = kde.evaluate(xs)
		
	mle = xs[np.argmax(kdes)]
	return mle



def beaumont_pmc_zipf_faster(ns, n_particles, survival_fraction, generations):

	n_seeds = int(n_particles/survival_fraction)
	min_theta=1.001
	max_theta=3
	n_data = sum(ns)

	# 1. Generate thetas from prior
	ds = []
	thetas = np.random.uniform(low=min_theta, high=max_theta, size=n_particles)
	for j in range(n_particles):
		theta_j = thetas[j]
		z_j = get_ranked_empirical_counts_from_infinite_power_law(theta_j, N=n_data)
		d_j = scipy.stats.wasserstein_distance(ns, z_j)
		ds.append(d_j)

	ws = np.full(n_particles, 1/n_particles)

	var = 2*np.var(thetas)
	sd = np.sqrt(var)
	#2. 

	for g in range(generations):
		print("Generation ", g)
		ps = ws/np.sum(ws)
		check_count = 0
		
		theta_stars = []
		d_stars = []

		# Generate theta_stars / seeds 
		for i in range(n_seeds):
			hit = False
			check_count += 1

			# "Hit" is just valid parameters
			while not hit:

				j = np.random.choice(len(thetas), p=ps)
				theta_j = thetas[j]
				theta_star = np.random.normal(loc=theta_j, scale=sd)

				if theta_star > min_theta and theta_star<max_theta:
					z_star = get_ranked_empirical_counts_from_infinite_power_law(theta_star, N=n_data)
					d_star = scipy.stats.wasserstein_distance(ns, z_star)
					theta_stars.append(theta_star)
					d_stars.append(d_star)
					hit = True


		# Get the next thetas, next ds and new tolerance
		# Pick the n_particles theta_stars with the lowest distances
		thetas_next, ds_next, tolerance = get_successful_stars(d_stars, theta_stars, n_particles)


		# Get the new weights
		ws_next = get_new_weights_uniform_prior(thetas_next, thetas, ws, sd)

		thetas = thetas_next
		ws = ws_next



		var = 2*np.cov(thetas, aweights=ws)
		sd= math.sqrt(var)
		
	# PLot kde
	kde = scipy.stats.gaussian_kde(thetas, weights=ws)
	xs = np.linspace(1,2.3, 10000)
	kdes = [kde.evaluate(x_i) for x_i in xs]

	mle = xs[np.argmax(kdes)]
	return mle


def zipf_test():

	np.random.seed(1)

	n_data = 10000
	exponent = 1.9
	n_particles = 256
	survival_fraction = 0.05
	generations = 5

	ns = get_ranked_empirical_counts_from_infinite_power_law(exponent, N=n_data)
	#mle = beaumont_pmc_zipf(ns, n_particles=256, survival_fraction=0.6, generations=10)
	mle = beaumont_pmc_zipf_faster(ns, n_particles, survival_fraction, generations)
	print("MLE IS ", mle)
	plt.axvline(mle)
	plt.show()

def zipf_experiment():

	results_filename = "zipf_beaumont_results_faster.csv"

	
	n_data = 10000
	n_particles = 256
	survival_fraction = 0.05
	generations = 5

	for seed in range(100):
		for n_particles in [128,256]:
			for survival_fraction in [0.05, 0.1]:			
				for exponent in np.linspace(1.01, 2, 10):
					
					np.random.seed(seed)
					
					ns = get_ranked_empirical_counts_from_infinite_power_law(exponent, N=n_data)
					
					
					start=time.time()
					mle = beaumont_pmc_zipf_faster(ns, n_particles, survival_fraction, generations)
					print("Seed {} exponent {} mle {}".format(seed, exponent, mle))
					end=time.time()
					csv_row = ["Beaumont 2007 Basic", seed, exponent, n_particles, survival_fraction, 
							n_data, generations, mle, end-start]
					
					append_to_csv(csv_row, results_filename)


def basic_test():

	seed = 5
	np.random.seed(seed)

	x = get_exponential_data(lamb=0.6, size=100)
	prior_alpha = 1
	prior_beta = 1
	#beaumont_pmc_gamma_prior(x, prior_alpha, prior_beta)
	beaumont_pmc_gamma_prior_dynamic_tolerance(x, prior_alpha, prior_beta)
	
	plt.title("Posterior WABC with Data from an Exponential Model\nGamma Prior Algo from Beaumont 2007")
	plt.legend()
	plt.xlabel("$\lambda$")
	plt.ylabel("$P(\lambda|D)$")
	plt.savefig("../plots/images/wabc_exponential_beaumont_2007_fast_{}.png".format(seed))

	plt.show()



if __name__=="__main__":
	zipf_experiment()