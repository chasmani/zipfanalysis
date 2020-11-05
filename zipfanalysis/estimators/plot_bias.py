
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_goffard():

	input_filename = "goffard_overnight_results_for_figure.csv"
	names = ["seed", "alpha", "method", "bw", "succ_prop", "gens", "n_particles", "alpha_hat", "time"]
	df = pd.read_csv(input_filename, names=names, sep=";")
	df = df[(df["method"] == "goffard") & (df["succ_prop"] == 0.1) & (df["gens"] == 4)]
	df["bias"] = df["alpha_hat"] - df["alpha"]
	df_means = df.groupby("alpha").mean().reset_index()
	sns.scatterplot(x="alpha", y="bias", data=df_means)
	plt.show()

def plot_beaumont():
	input_filename = "zipf_beaumont_results.csv"
	names = ["method", "seed", "exponent", "n_particles", "survival_fraction", 
					"n_data", "generations", "mle", "time"]
	df = pd.read_csv(input_filename, names=names, sep=";")
	df = df[(df["method"] == "Beaumont 2007 Basic")]
	df["bias"] = df["mle"] - df["exponent"]
	df_means = df.groupby("exponent").mean().reset_index()
	#sns.scatterplot(x="exponent", y="bias", data=df)

	df_vars = df.groupby("exponent").var().reset_index()
	sns.lineplot(x="exponent", y="bias", data=df_vars)

	plt.show()	

def plot_beaumont_faster():

	input_filename = "zipf_beaumont_results_faster.csv"
	names = ["method", "seed", "exponent", "n_particles", "survival_fraction", 
					"n_data", "generations", "mle", "time"]
	df = pd.read_csv(input_filename, names=names, sep=";")
	survival_fraction = 0.05
	n_particles = 256
	df = df[(df["method"] == "Beaumont 2007 Basic") & (df["survival_fraction"] == survival_fraction) & (df["n_particles"] == n_particles)]
	df["bias"] = df["mle"] - df["exponent"]
	
	"""
	# Bias
	df_means = df.groupby("exponent").mean().reset_index()
	sns.lineplot(x="exponent", y="bias", data=df, ci=68)
	plt.title("Bias of Beaumont Faster Algo\n alpha={}, no. particles={}".format(survival_fraction, n_particles))
	plt.savefig("../plots/images/beaumont_fast_bias_sf_{}_n_parts_{}.png".format(survival_fraction, n_particles))
	"""

	# Variance
	df_vars = df.groupby("exponent").var().reset_index()
	sns.lineplot(x="exponent", y="bias", data=df_vars)
	plt.title(r"Variance of Beaumont Faster Algo\n alpha={}, no. particles={}".format(survival_fraction, n_particles))
	plt.savefig("../plots/images/beaumont_fast_variance_sf_{}_n_parts_{}.png".format(survival_fraction, n_particles))
	

	plt.show()					


plot_beaumont_faster()