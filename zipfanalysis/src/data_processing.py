

import re
from collections import Counter

import nltk 


def read_txt_file(filepath):

	with open(filepath, "r") as file:
		text = file.read()
	return text

def clean_text(text):

	# Remove apostrophes to handle contractions
	# it's -> its
	# can't -> cant
	text = re.sub(r"\s\'|\'", "", text)
	return text

def convert_text_to_word_list(text):
	# nltk's tokenize used to extract words
	tokens = nltk.tokenize.word_tokenize(text)
	
	# Get lowercase versions of words
	# No punctuation or numbers included
	words = [word.lower() for word in tokens if word.isalpha()]
	
	return words


def convert_word_list_to_frequency_dist(words):

	word_counts = Counter(words)
	frequency_dist = [v for k,v in word_counts.most_common()]
	return frequency_dist


def convert_frequency_dist_to_frequency_counts(frequency_dist):
	"""
	Input a frequency distribution of how many words of each rank appeared 
	frequency_dist[i] = # of occurences of ith ranked word
	Outputs a frequency counts distribution
	frequency_counts[j] = # of tokens that have a frequency count of j
	"""
	frequency_counts = []
	for f_i in range(1, max(frequency_dist)+1):
		frequency_counts.append(frequency_dist.count(int(f_i)))
	return frequency_counts


def convert_frequency_counts_to_frequency_dist(frequency_counts):
	"""
	Input a a frequency counts distribution
	frequency_counts[j] = # of tokens that have a frequency count of j
	Output a frequency distribution of how many words of each rank appeared 
	frequency_dist[i] = # of occurences of ith ranked word
	"""
	frequency_dist = []
	for frequency in range(1, len(frequency_counts)+1):
		frequency_dist += [frequency] * frequency_counts[frequency-1]
	frequency_dist.sort(reverse=True)
	return frequency_dist