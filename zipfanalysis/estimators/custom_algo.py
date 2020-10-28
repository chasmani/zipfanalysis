
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


	
	ls = np.array(ls)/sum(ls)

	bar_width = 0.6/100

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

	
def custom_algo(ns, min_exponent=0.1, max_exponent=2.4, n_particles = 128, survival_fraction = 0.05):

	theta_0s = np.random.uniform(low=min_exponent, high=max_exponent, size=n_particles)
	n_data = len(ns)

	for run_count in range(10):

		theta_0s = np.random.uniform(low=min_exponent, high=max_exponent, size=n_particles)
		ds = []
		for k in range(n_particles):
			theta_k = theta_0s[k]
			z_k = get_exponential_data(theta_k, n_data)
			d_k = scipy.stats.wasserstein_distance(ns, z_k)
			ds.append(d_k)

		#1C, 2A and 2B Select tolerance and successful particles
		a_k, d_k, epsilon_k = extract_successful_trials(theta_0s, ds, survival_fraction*n_particles)

		min_exponent = max(min(a_k) - np.std(a_k), 0.1)
		max_exponent = max(a_k) + np.std(a_k)
		print(min_exponent, max_exponent)

	n_samples_abc_rejection = 2048
	theta_0s = np.random.uniform(low=min_exponent, high=max_exponent, size=n_particles)
	succesful_thetas = []

	for k in range(n_particles):
		theta_k = theta_0s[k]
		z_k = get_exponential_data(theta_k, n_data)
		d_k = scipy.stats.wasserstein_distance(ns, z_k)
		if d_k < epsilon_k:
			succesful_thetas.append(theta_k)
	sns.kdeplot(succesful_thetas, label="custom")


def basic_experiment():

	np.random.seed(5)

	d = get_exponential_data(lamb=0.8, size=200)
	custom_algo(d)
	plot_likelihood_function(d)
	
	plt.title("Posterior Custom with Data from an Exponential Model")
	plt.legend()
	plt.savefig("../plots/images/custom_algo.png")

	plt.show()




if __name__=="__main__":
	basic_experiment()