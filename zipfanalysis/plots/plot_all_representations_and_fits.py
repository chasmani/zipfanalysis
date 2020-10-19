
from zipfanalysis.preprocessing.preprocessing import get_rank_frequency_from_text

def get_ccdf_point_per_word(n):

	print(n)




def plot_ccdf():
	book = "../resources/books/alice_wonderland.txt"
	n = get_rank_frequency_from_text(book)
	get_ccdf_point_per_word(n)

if __name__=="__main__":
	plot_ccdf()