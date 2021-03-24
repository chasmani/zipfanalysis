

import numpy as np
from scipy.special import zeta
from scipy.optimize import bisect 
from scipy import optimize

# for testing

import plfit


def clauset_estimator(ns):
	"""
	Finds the root of the differential of the likelihood function
	This method is more than an order of magnitude faster than maximising the likelihood function
	
	Assumes an ns vector of counts of observations of event i like ns[i]
	"""
	return clauset_differential_root(ns)


################################################
# VERSION 1 - MAXIMISING LIKELIHOOD
def clauset_likelihood(lamb, ns):
	"""
	log L = -lamb* SUM(n(i) * ln(i)) - N * ln(zeta(lamb))
	"""

	sum_part = 0
	for j in range(len(ns)):
		rank = j+1
		sum_part += ns[j]*np.log(rank)

	z = zeta(lamb)
	log_l = -lamb * sum_part - sum(ns) * np.log(z)
	return log_l


def clauset_maximise_likelihood(ns):

	f = clauset_likelihood
	args = (ns, )
	x0 = 1 # initial guess

	mle = optimize.fmin(lambda x, ns: -f(x, ns), x0, args=args, xtol=1e-7)
	return mle[0]


#################################################
# VERSION 2 - ROOT OF DIFF LIKELIHOOD

def D_l(alpha, t):
	return log_deriv_zeta(alpha) + t	


def get_t(ns, x_min=1):
	sum_part = 0
	for j in range(len(ns)):
		rank = j+1
		sum_part += ns[j]*np.log(rank)
	return sum_part/sum(ns)


def log_zeta(alpha):
    return np.log(zeta(alpha, 1))


def log_deriv_zeta(alpha):
    h = 1e-5
    return (log_zeta(alpha+h) - log_zeta(alpha-h))/(2*h)


def clauset_differential_root(ns):
	t = get_t(ns)
	a,b = 1.01, 10
	alpha_hat = bisect(D_l, a, b, args=t, xtol=1e-7)
	return alpha_hat


##################################################
# FOR TESTING - Using powerlaw Package
def powerlaw_package_clauset_estimator(ns):
	# Convert to x vector to use powerlar library
	x = []
	for i in range(len(ns)):
		rank = i+1
		x += [rank]*ns[i]
	lib_fit = powerlaw.Fit(x, discrete=True, xmin=1, estimate_discrete=False)
	
	lib_alpha =  lib_fit.power_law.alpha
	return lib_alpha



if __name__=="__main__":
	pass