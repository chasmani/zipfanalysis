
import matplotlib.pyplot as plt

import src.data_processing as data_processing


class ZipfAnalysis:
	"""
	Analysis of one data set i.e. one book
	"""
	DATA_TYPES = [
			"input_filename",
			"text_string",
			"word_list",
			"frequency_dist",
			"frequency_counts"
			]

	def __init__(self, data=None, data_type=None):
		"""
		
		"""
		if data == None or data_type not in self.DATA_TYPES:
			raise ValueError("""
				The Analysis object requires 
					1) some data to work on, data= <the data>
					2) The data_type, data_type="<string>" 
				
				Acceptable data_types and examples:

				INSERT EXAMPLES HERE
				"""
				)

		self.data = data
		self.data_type = data_type		

		# Initialise all the data representations
		self.input_filename = None
		self.text_string = None
		self.word_list = None
		self.frequency_dist = None
		self.frequency_counts = None

		self.import_data(data, data_type)

		self.process_data()


	def import_data(self, data, data_type):

		if data_type == "input_filename":
			self.input_filename = data		

		if data_type == "text_string":
			self.text_string = data

		if data_type == "word_list":
			self.word_list = data

		if data_type == "frequency_dist":
			self.frequency_dist = list(map(int, data))

		if data_type == "frequency_counts":
			self.frequency_counts = list(map(int, data))


	def process_data(self):
		"""
		Process data, fill in all the data representations that can be filled in
		given the data_type input
		"""

		if self.input_filename:
			self.text_string = data_processing.read_txt_file(self.input_filename)

		if self.text_string:
			clean_text = data_processing.clean_text(self.text_string)
			self.word_list = data_processing.convert_text_to_word_list(clean_text)

		if self.word_list:
			self.frequency_dist = data_processing.convert_word_list_to_frequency_dist(self.word_list)

		if self.frequency_dist:
			self.frequency_counts = data_processing.convert_frequency_dist_to_frequency_counts(self.frequency_dist)

		if self.frequency_counts and not self.frequency_dist:
			self.frequency_dist = data_processing.convert_frequency_counts_to_frequency_dist(self.frequency_counts)


	def plot_rank_frequency_pdf(self):

		plt.scatter(range(1, len(self.frequency_dist)+1), self.frequency_dist)
		plt.xlabel("Rank")
		plt.ylabel("Frequency")		
		plt.xscale("log")
		plt.yscale("log")


	def plot_frequency_counts_pdf(self):

		# Don't plot zeroes in the frequency_counts (avoid issues with log(0))
		# Just get non-zero values
		ff = []
		cc = []
		for f_i in range(1, len(self.frequency_counts)+1):
			if self.frequency_counts[f_i-1] > 0:
				ff.append(f_i)
				cc.append(self.frequency_counts[f_i-1])

		plt.scatter(ff, cc)
		plt.xlabel("Frequency")
		plt.ylabel("Count")		
		plt.xscale("log")
		plt.yscale("log")





if __name__=="__main__":
	za = ZipfAnalysis([1, 2, 2, 1], data_type="frequency_dist")
	za = ZipfAnalysis("data/books/ulysses.txt", data_type="input_filename")
	za.plot_frequency_counts_pdf()
	plt.show()