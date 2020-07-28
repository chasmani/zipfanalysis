
import re
import unidecode

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

