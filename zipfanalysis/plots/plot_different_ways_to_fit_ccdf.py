
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


def get_ccdf_point_per_word(x):


	xx_line = np.linspace(0, 4, 1000)
	yy_line = np.array([(x>val).sum() for val in xx_line])/len(x)

	ax1 = plt.subplot(1,3,1)

	plt.ylabel("P(X > x)")
	plt.suptitle("Complementary CDF for data x={}".format(x))

	ccdf_all_points = np.linspace(1, 1/len(x), len(x))
	plt.scatter(x, ccdf_all_points, color="red")
	plt.plot(xx_line, yy_line)

	plt.title("A. All points")

	# Fit a model to the data
	sm_xs = sm.add_constant(x)
	model = sm.OLS(ccdf_all_points, sm_xs)
	results = model.fit()
	yy_fit = [results.params[0] + results.params[1]*val for val in xx_line]
	plt.plot(xx_line, yy_fit, color="red")



	

	ax2 = plt.subplot(1,3,2, sharey=ax1)

	plt.xlabel("x")
	
	counts = Counter(x)
	ns = []

	for k,v in counts.most_common():
		ns.append(v)

	print(ns)

	ccdf_n_xs = np.arange(1, len(ns)+1)
	ccdf_n_ys_end = 1 - np.cumsum(ns)/np.sum(ns)
	print(ccdf_n_ys_end)
	plt.scatter(ccdf_n_xs, ccdf_n_ys_end, color="red")
	plt.title("B. Bottom of steps")
	plt.plot(xx_line, yy_line)

	# Fit a model to the data
	sm_xs = sm.add_constant(ccdf_n_xs)
	model = sm.OLS(ccdf_n_ys_end, sm_xs)
	results = model.fit()
	yy_fit = [results.params[0] + results.params[1]*val for val in xx_line]
	plt.plot(xx_line, yy_fit, color="red")

	ax3 = plt.subplot(1,3,3, sharey=ax1)

	ccdf_n_ys_starts = [np.sum(ns[i:]) for i in range(len(ns))]/np.sum(ns)
	print(ccdf_n_ys_starts)
	plt.scatter(ccdf_n_xs, ccdf_n_ys_starts, color="red")
	plt.title("C. Top of steps")

	plt.plot(xx_line, yy_line)

	# Fit a model to the data
	sm_xs = sm.add_constant(ccdf_n_xs)
	model = sm.OLS(ccdf_n_ys_starts, sm_xs)
	results = model.fit()
	yy_fit = [results.params[0] + results.params[1]*val for val in xx_line]
	plt.plot(xx_line, yy_fit, color="red")

	plt.savefig("images/different_types_of_ccdf.png")

	plt.show()
	

if __name__=="__main__":
	x = [1,1,1,2,3]
	get_ccdf_point_per_word(x)