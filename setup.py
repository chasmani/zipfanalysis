import setuptools

def readme():

	with open("README.rst", "r") as fh:
		return fh.read()

print(setuptools.find_packages())


setuptools.setup(
	name = "zipfanalysis",
	version = "0.4",
	author = "Charlie Pilgrim",
	author_email = "pilgrimcharlie2@gmail.com",
	description = "Tools for analysing Zipf's law from text samples",
	long_description = readme(),
	long_description_content_type="text/x-rst",
	url = "https://github.com/chasmani/zipfanalysis",
	python_requires='>=3.6.9',	
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
		"Topic :: Text Processing :: Linguistic"
	],
	packages = setuptools.find_packages(),
	)