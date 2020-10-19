

import pandas as pd
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import numpy as np
import scipy

df = pd.read_csv('ks_test_data_2.csv', delimiter="|")
data1 = df.iloc[:,0]
data2 = df.iloc[:,1]
test = ks_2samp(data1,data2)
print(test[0])
print(test)



d1 = [1,1,2,2,3,3,4,4,5,5]
n1 = [2,2,2,2,2]
n2 = [0,0,6,1,3]

F1 = np.cumsum(n1)
print(F1)

ns = [3,4,1,3,2]
x = []
for i in range(1, len(ns)+1):
	x += [i]*ns[i-1]
print(x)


a= 1.3
loc = -10

x = np.arange(scipy.stats.zipf.ppf(0.01, a, loc),
              scipy.stats.zipf.ppf(0.9, a, loc))
plt.plot(x, scipy.stats.zipf.pmf(x, a, loc), 'bo', ms=8, label='zipf pmf')
plt.axhline(1)
plt.xscale("log")
plt.yscale("log")

plt.show()

N= 10000
x = scipy.stats.zipf.rvs(a, loc=loc, size=N)

x = x[x>0]

print(x)
print(len(x))

N_extra_needed_plus_few_more = ((N/len(x) - 1) * N)*1.2
print(N_extra_needed_plus_few_more)
x_2 = scipy.stats.zipf.rvs(a, loc=loc, size=int(N_extra_needed_plus_few_more))
x_2 = x_2[x_2>0]

print(x)
print(x_2)

x_all = np.concatenate((x_2, x))
print(x_all)
print(len(x_all))

if (len(x_all) > N):
	x = 