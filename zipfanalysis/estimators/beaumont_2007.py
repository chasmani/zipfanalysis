
import math

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.special
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



def beaumont_pmc(x):

	
	n_particles = 1024
	survival_fraction = 0.3
	min_lamb=0.1
	max_lamb=2
	n_data = len(x)
	generations = 10

	# 1. Generate thetas from prior
	ds = []
	thetas = np.random.uniform(low=min_lamb, high=max_lamb, size=n_particles)
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
		test_count = 0
		for i in range(n_particles):
			hit=False
			while not hit:
				test_count += 1
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

		ws_next = []

		for i in range(n_particles):
			prior_i = 1
			stand_norm_var = (thetas_next[i] - thetas[j]) / sd
			w_i = prior_i/(ws_sum) * scipy.stats.norm.pdf(stand_norm_var)
			ws_next.append(w_i)

		thetas = thetas_next
		ds = ds_next

		plt.scatter(thetas, ds)
		plt.show()

		ws = ws_next
		var = 2*np.var(thetas)
		sd= math.sqrt(var)
		ws_sum = np.sum(ws)
		print(g, tolerance, sd, test_count)
		plot_likelihood_function(x)
	
		# PLot kde
		"""
		kde = scipy.stats.gaussian_kde(thetas, weights=ws)
		xs = np.linspace(0.4,0.8)
		kdes = [kde.evaluate(x_i) for x_i in xs]
		plt.plot(xs, kdes, label="WABC")
		"""

		sns.kdeplot(thetas, label=g)

		plt.show()

def plot_posterior_exponential(x, prior_alpha, prior_beta):

	posterior_alpha = prior_alpha + len(x)
	posterior_beta = prior_beta + sum(x)

	print(posterior_alpha)
	xs = np.linspace(0.01,2, 200)
	ps = [get_gamma_p(x, posterior_alpha, posterior_beta) for x in xs]

	plt.plot(xs, ps, label="Actual Posterior")

def get_gamma_p(x, alpha, beta):

	p = (beta**alpha) /scipy.special.gamma(alpha) * x**(alpha-1) * np.exp(-1*beta*x)
	return p

def beaumont_pmc_gamma_prior(x, prior_alpha, prior_beta):

	n_particles = 256
	survival_fraction = 0.5
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
		for i in range(n_particles):
			hit=False
			while not hit:
				test_count += 1
				print(test_count)
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
						prior_i = get_gamma_p(theta_prime, prior_alpha, prior_beta)						
						summation = beaumont_sum(ws, sd, theta_prime, thetas)
						w_i = prior_i/summation
						ws_next.append(w_i)						

		thetas = thetas_next
		ds = ds_next

		plt.scatter(thetas, ds)
		plt.show()

		ws = ws_next
		var = 2*np.cov(thetas, aweights=ws)
		sd= math.sqrt(var)
		ws_sum = np.sum(ws)
		print(g, tolerance, sd, test_count)
		plot_posterior_exponential(x, prior_alpha, prior_beta)
	
		# PLot kde
		kde = scipy.stats.gaussian_kde(thetas, weights=ws)
		xs = np.linspace(0.4,0.8)
		kdes = [kde.evaluate(x_i) for x_i in xs]
		plt.plot(xs, kdes, label="WABC")
		
		#sns.kdeplot(thetas, label=g)

		plt.show()


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



def basic_test():

	np.random.seed(9)

	x = get_exponential_data(lamb=0.6, size=100)
	prior_alpha = 1
	prior_beta = 1
	beaumont_pmc_gamma_prior(x, prior_alpha, prior_beta)
	#plot_likelihood_exponential(x, prior_alpha, prior_beta)
	plot_posterior_exponential(x, prior_alpha, prior_beta)
	
	plt.title("Posterior WABC with Data from an Exponential Model\nBeaumont 2006")
	plt.legend()
	#plt.savefig("../plots/images/wabc_exponential_simple_kernel.png")

	plt.show()

def sanity_check():

	x = get_exponential_data(lamb=0.6, size=200)
	z_1 = get_exponential_data(lamb=0.62, size=200)
	z_2 = get_exponential_data(lamb=0.9, size=200)
	d_1 = scipy.stats.wasserstein_distance(x, z_1)
	d_2 = scipy.stats.wasserstein_distance(x, z_2)
	print(d_1, d_2)



if __name__=="__main__":
	basic_test()