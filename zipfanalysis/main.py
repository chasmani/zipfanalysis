
import matplotlib.pyplot as plt

import src.data_processing as data_processing
import src.mle as mle


class ZipfAnalysis:
	"""
	Analysis of one data set i.e. one book
	"""
	DATA_TYPES = [
			"input_filename",
			"text_string",
			"word_list",
			"rank_frequency",
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
		self.rank_frequency = None
		self.frequency_counts = None

		self.rank_frequency_ols = None
		self.rank_frequency_mle = None
		self.frequency_counts_ols = None
		self.frequency_counts_mle = None

		self.import_data(data, data_type)

		self.process_data()


	def import_data(self, data, data_type):

		if data_type == "input_filename":
			self.input_filename = data		

		if data_type == "text_string":
			self.text_string = data

		if data_type == "word_list":
			self.word_list = data

		if data_type == "rank_frequency":
			self.rank_frequency = list(map(int, data))

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
			self.rank_frequency = data_processing.convert_word_list_to_rank_frequency(self.word_list)

		if self.rank_frequency:
			self.frequency_counts = data_processing.convert_rank_frequency_to_frequency_counts(self.rank_frequency)

		if self.frequency_counts and not self.rank_frequency:
			self.rank_frequency = data_processing.convert_frequency_counts_to_rank_frequency(self.frequency_counts)

	def get_rank_frequency_ols(self):
		pass

	def get_frequency_counts_ols(self):
		pass

	def get_rank_frequency_mle(self):
		if not self.rank_frequency_mle:
			self.rank_frequency_mle = mle.clauset_estimator(self.rank_frequency)
		return self.rank_frequency_mle

	def get_frequency_counts_mle(self):
		if not self.frequency_counts_mle:
			self.frequency_counts_mle = mle.clauset_estimator(self.frequency_counts)
		return self.frequency_counts_mle


	def plot_rank_frequency_pdf(self, **kwargs):

		rr = range(1, len(self.rank_frequency)+1)
		nn = self.rank_frequency

		plt.scatter(rr, nn, **kwargs)
		plt.xlabel("Rank")
		plt.ylabel("Frequency")		
		plt.xscale("log")
		plt.yscale("log")


	def plot_rank_frequency_ols(self):
		pass

	def plot_rank_frequency_mle(self):
		pass

	def plot_frequency_counts_pdf(self, **kwargs):

		# Don't plot zeroes in the frequency_counts (avoid issues with log(0))
		# Just get non-zero values
		nn = []
		cc = []
		for n_i in range(1, len(self.frequency_counts)+1):
			if self.frequency_counts[n_i-1] > 0:
				nn.append(n_i)
				cc.append(self.frequency_counts[n_i-1])

		plt.scatter(nn, cc, **kwargs)
		plt.xlabel("Frequency")
		plt.ylabel("Count")		
		plt.xscale("log")
		plt.yscale("log")

	def plot_frequency_counts_ols(self):
		pass

	def plot_frequency_counts_mle(self):
		pass



if __name__=="__main__":
	za = ZipfAnalysis([1, 2, 2, 1], data_type="rank_frequency")
	za = ZipfAnalysis("data/books/ulysses.txt", data_type="input_filename")
	za.plot_frequency_counts_pdf(s=5)
	
	print(za.get_rank_frequency_mle())
	print(za.get_frequency_counts_mle())

	plt.show()