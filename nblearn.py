#!/usr/bin/python

import collections
import argparse
import os
from os import path
import json
import re


class NaiveBayesLearn:
	def __init__(self):
		self.spamFiles = []
		self.hamFiles = []
		self.szVocabulary = 0
		self.model = {}
		self.portion = None
		self.hamCollection = collections.defaultdict(int)
		self.spamCollection = collections.defaultdict(int)

	def getszVocabulary(self):
		repeated_words =  len(list(filter(lambda x: x in self.hamCollection.keys(), self.spamCollection.keys())))
		return (len(self.hamCollection.keys()) + len(self.spamCollection.keys()) - repeated_words)

	def appendList(self, category, rootdir, txtfile):
		if category:
			self.spamFiles.append(os.path.join(rootdir,txtfile))
		else:
			self.hamFiles.append(os.path.join(rootdir,txtfile)) 

	def segregateFiles(self, dirname, portion = None):
		isSpam = re.compile("spam")
		for root, dirs, files in os.walk(dirname):
			for f in files:
				category = isSpam.match(os.path.basename(root))
				self.appendList(category, root, f)
		if portion:
			#customize the training data size
			hamfile_listlength = len(self.hamFiles)
			spamfile_listlength = len(self.spamFiles)
			hamlistredux = (portion * hamfile_listlength)//100
			spamlistredux = (portion * spamfile_listlength)//100
			#reduce the list length to desired size
			self.spamFiles = self.spamFiles[:spamlistredux]
			self.hamFiles = self.hamFiles[:hamlistredux]			
	
	def categorizeTokens(self, files, vocab):
		for _file in files:
			with open(_file,"r", encoding="latin1") as _fhandle:
				content = _fhandle.read()
				_fhandle.close()
			tokens = content.split()
			for t in tokens:
				vocab[t] += 1

	def generateFeaturesforTokens(self, category):
		# read each file from the processed list and extract features
		# update the feature counts into the hash based on the category
		if category == "ham":
			files = self.hamFiles
			collection = self.hamCollection
			self.categorizeTokens(files, collection)
		else:
			files = self.spamFiles
			collection = self.spamCollection
			self.categorizeTokens(files, collection)
			
		
	def generateModel(self):
		documents = len(self.hamFiles) + len(self.spamFiles)
		self.model["Documents"] = documents
		spams = len(self.spamFiles)
		self.model["TotalSpamFiles"] = spams
		hams = len(self.hamFiles)
		self.model["TotalHamFiles"] = hams
		spamwords = sum(self.spamCollection.values())
		self.model["Spams"] = spamwords
		hamwords = sum(self.hamCollection.values())
		self.model["Hams"] = hamwords
		self.model["|V|"] = self.getszVocabulary()
		self.model["HamTokens"] = self.hamCollection
		self.model["SpamTokens"] = self.spamCollection
		# write model data to file
		if self.portion:
			output_filename = "nbmodel_"+str(self.portion)+".txt" 
		else:
			output_filename = "nbmodel.txt"
		with open(output_filename,"w+") as _filehandle:
			json.dump(self.model,_filehandle, indent=4)
			_filehandle.close()



parser = argparse.ArgumentParser(usage="python nblearn.py DIRECTORY", description="Build a naive bayes model")
parser.add_argument('idir', help="inputdir help")
parser.add_argument("-d","--portion", help="Specify training portion percentage",type=int)
args = parser.parse_args()
nbLearn = NaiveBayesLearn()
if args.portion:
	nbLearn.portion = args.portion
nbLearn.segregateFiles(args.idir,args.portion)
nbLearn.generateFeaturesforTokens("spam")
nbLearn.generateFeaturesforTokens("ham")
nbLearn.generateModel()