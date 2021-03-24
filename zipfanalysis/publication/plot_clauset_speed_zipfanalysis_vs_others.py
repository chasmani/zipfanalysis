
import time

import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas as pd
import powerlaw
import matplotlib.pyplot as plt

import src.clauset as clauset
from main import ZipfAnalysis

from general_utilities import append_to_csv


RESULTS_HEADER = ["book", "n_words", "n_trials", "package", "time"]

books = [
			"alice_wonderland.txt",
			"chronicle_london.txt",
			"moby_dick.txt",
			"tale_two_cities.txt",
			"ulysses.txt"
		]

book_colors = {	
			"alice_wonderland.txt":"blue",
			"chronicle_london.txt":"red",
			"moby_dick.txt":"green",
			"tale_two_cities.txt":"pink",
			"ulysses.txt":"black"
}


def analyse_speed_vs_powerlaw():

	results_csv = "results/speedtest.csv"


	

	n_trials = 10

	for book in books:
		print("Testing ", book)
		book_filepath = "../data/books/{}".format(book)
		za = ZipfAnalysis(data=book_filepath, data_type="input_filename")
		ns = za.frequency_dist
		

		start_time_za = time.time()
		for i in range(n_trials):
			clauset.clauset_estimator(ns)
		end_time_za = time.time()

		time_za = (end_time_za-start_time_za)/n_trials
		print("ZA: ", time_za)

		x = []
		for i in range(len(ns)):
			rank = i+1
			x += [rank]*ns[i]
	

		start_time_powerlaw = time.time()
		for i in range(n_trials):
			lib_fit = powerlaw.Fit(x, discrete=True, xmin=1, estimate_discrete=False)			
		end_time_powerlaw = time.time()

		time_powerlaw = (end_time_powerlaw-start_time_powerlaw)/n_trials
		print("Powerlaw: ", time_powerlaw)

		L = sum(ns)

		csv_row = [book, L, n_trials, "zipfanalysis", time_za]

		append_to_csv(csv_row, results_csv)
		csv_row = [book, L, n_trials, "powerlaw", time_powerlaw]
		
		append_to_csv(csv_row, results_csv)

def plot_speed_test():

	results_csv = "results/speedtest.csv"

	df = pd.read_csv(results_csv, delimiter=";", names=RESULTS_HEADER)

	df = df[(df['n_trials'] == 10)]


	for book in books:

		for package in ["zipfanalysis", "powerlaw"]:
			this_df = df[(df['book'] == book) & (df['package'] == package)]

			book_label=book.replace(".txt", "").replace("_", " ").title()

			if package == "zipfanalysis":
				label=book_label
				marker = "o"
			else:
				label=None
				marker = "s"



			plt.scatter(this_df["n_words"], this_df["time"], label=label, color=book_colors[book], marker=marker)


	plt.yscale("log")

	plt.xlabel("Book Length (words)")
	plt.ylabel("Clauset Estimate Time")

	plt.legend()

	plt.savefig("images/speed_test.png")

	plt.show()

		
if __name__=="__main__":
	plot_speed_test()