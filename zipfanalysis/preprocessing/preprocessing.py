#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import unidecode
from collections import Counter

def clean_text(content):
	"""
	Take a text sample, extract only letters, spaces and full stops
	Also some processing of things like ampersands
	"""

	# Convert to ascii 
	content = unidecode.unidecode(content)

	# 1. Replace all end of sentence punctuation with a full stop	
	content = re.sub(r'[\?\!]', '.', content)

	#2. Collapse any repeated full stops to just one full stop
	content = re.sub(r'\.(\s*\.)*', ' . ', content)

	# Convert ampersands to ands
	content = re.sub(r'\&', 'and', content)

	# Convert hyphens and underscores to spaces
	content = re.sub(r'[\—\-\_]', ' ', content)

	# Remove all other punctuation
	content = re.sub(r'[^\w\d\s]', '', content)

	# Replace any numbers with #
	content = re.sub(r'\d+', '#', content)

	# Make all lowercase
	content = content.lower()

	# Remove repeated whitespace
	content = re.sub(r'\s{1,}', ' ', content)

	# Remove last whitespace
	content = re.sub(r'\s{1,}$', '', content)	

	return content


def get_word_counts(input_filename):
	"""
	Given an input filename, 
		open that file
		Preprocess/clean the text
		Count the word occurences and 
		Return a Counter object 
	"""

	with open (input_filename, 'r' ) as f:
		content = f.read()
		clean_content = clean_text(content)

	clean_words = clean_content.split(" ")

	# Data structure to count the ngrams
	word_counts = Counter(clean_words)
	return word_counts


def get_rank_frequency_from_text(input_filename):
	"""
	Given an input filename
	Process the text, count the word occurences
	Return an empirical ranking vector n where n[k] is the count of the kth most common word
	"""

	word_counts = get_word_counts(input_filename)
	n = [v for k,v in word_counts.most_common()]
	return n
