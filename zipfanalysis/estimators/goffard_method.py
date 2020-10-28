
import math
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kde import KDEUnivariate
import scipy.stats

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
		print(ws)

		kde = scipy.stats.gaussian_kde(a_ks, weights=ws)




	xs = np.linspace(2,6)
	kdes = [kde.evaluate(x_i) for x_i in xs]
	plt.plot(xs, kdes, label="WABC")


def extract_successful_trials(parameters, distances, alpha):
	"""
	Get close trial results, at least as many as "required_accepted_parameters"
	"""
	# Choose a low tolerance to begin
	tolerance = 0.001
	finished = False
	required_accepted_parameters = alpha * len(parameters)
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





def test_goffard():

	np.random.seed(1)
	prior_mean = 0
	prior_variance = 20
	known_variance = 10
	actual_mean = 4
	x = np.random.normal(loc=actual_mean, scale=math.sqrt(known_variance), size=100)
	goffard_normal(x, known_variance, prior_mean, prior_variance)
	plot_likelihood_normal(x, known_variance, prior_mean, prior_variance)	
	plt.show()
	

if __name__=="__main__":
	test_goffard()