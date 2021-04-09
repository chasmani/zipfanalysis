

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


def convert_word_list_to_rank_frequency(words):

	word_counts = Counter(words)
	rank_frequency = [v for k,v in word_counts.most_common()]
	return rank_frequency


def convert_rank_frequency_to_frequency_counts(rank_frequency):
	"""
	Input a frequency distribution of how many words of each rank appeared 
	rank_frequency[i] = # of occurences of ith ranked word
	Outputs a frequency counts distribution
	frequency_counts[j] = # of tokens that have a frequency count of j
	"""
	frequency_counts = []
	for f_i in range(1, max(rank_frequency)+1):
		frequency_counts.append(rank_frequency.count(int(f_i)))
	return frequency_counts


def convert_frequency_counts_to_rank_frequency(frequency_counts):
	"""
	Input a a frequency counts distribution
	frequency_counts[j] = # of tokens that have a frequency count of j
	Output a frequency distribution of how many words of each rank appeared 
	rank_frequency[i] = # of occurences of ith ranked word
	"""
	rank_frequency = []
	for frequency in range(1, len(frequency_counts)+1):
		rank_frequency += [frequency] * frequency_counts[frequency-1]
	rank_frequency.sort(reverse=True)
	return rank_frequency