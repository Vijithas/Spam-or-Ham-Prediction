#!/usr/bin/python

import os
import sys
import math
import argparse
from os import path
import re
import json

#Class to categorize the email as spam or ham using naive bayes
class NaiveBayesClassify:
	def __init__(self):
		#declare and initialize the variables of the class
		self.testlist = []
		self.CategoryResults = []
		self.model = {}
		self.PSpam = 1
		self.PHam = 1
		self.PredictedHams = []
		self.PredictedSpams = []
		self.TestDataHams = []
		self.TestDataSpams = []

	#Helper method to append items to list
	#Used to add filenames and values to their respective list 
	def appendList(self, list, item):
		list.append(item)

	# Method to compute probability value of the class
	def computeclassprobability(self):
		self.PSpam = self.model["TotalSpamFiles"]/self.model["Documents"]
		self.PHam = self.model["TotalHamFiles"]/self.model["Documents"]

	# Method to read the data from the model
	def readmodeldata(self, fname):
		try:
			with open(fname,"r") as _filehandle:
				self.model = json.load(_filehandle)
				_filehandle.close()
				self.computeclassprobability()

		except FileNotFoundError:
			print("Cannot find the model file")
			sys.exit()

	# Method to read the test data
	def readTestData(self, dirname):
		for root, dirs, files in os.walk(dirname):
			for f in files:
				self.appendList(self.testlist, os.path.join(root,f))
	
	# Method to rretrive the count of each token from the model file
	def getTokenCount(self, token, category):
		if token in self.model[category]:
			count = self.model[category][token]
		else:
			#return count as zero if the model does not contain any information
			count =  0		
		return count

	# method to calculate the probability of each word in the file
	def wordprobabilityLog(self, token, category):
		if category == "ham":
			count = self.getTokenCount(token, "HamTokens")
			return math.log(( count  + 1)/(self.model["Hams"] + self.model["|V|"])) 
		elif category == "spam":
			count = self.getTokenCount(token, "SpamTokens")
			return math.log(( count + 1)/(self.model["Spams"] + self.model["|V|"]))

	# Method to categorize the file as ham or spam based on probability values
	def Classifier(self, PSpam, PHam, fname):
		if PSpam < PHam:
			self.appendList(self.PredictedHams, fname)
			return "HAM "+ fname
		else:
			self.appendList(self.PredictedSpams, fname)
			return "SPAM "+ fname

	# read a file -> tokenize -> calculate probability spam and ham
	def filecategorization(self, fname):
		PMsgHam = 0
		PMsgSpam = 0
		doc_classification = None
		try:
			with open(fname,"r",encoding="latin1") as _filehandle:
				content = _filehandle.read()
				_filehandle.close()
			tokens = content.split()
			for token in tokens:
				PMsgSpam = PMsgSpam + self.wordprobabilityLog(token,"spam")
				PMsgHam = PMsgHam + self.wordprobabilityLog(token,"ham")
				
			probabilityofSpam = math.log(self.PSpam)
			probabilityofHam = math.log(self.PHam)

			PDocSpam = probabilityofSpam + PMsgSpam
			PDocHam = probabilityofHam + PMsgHam
			
			result = self.Classifier(PDocSpam, PDocHam, fname)
			
			return result
		except Exception as err:
			raise Exception(err)

	# Write the output to a file in the following format:
	# LABEL1 path1...
	# LABEL2 path2...
	def categorizeData(self):
		#isSpam = re.compile(".*.spam.txt")
		#self.TestDataSpams = [spam for spam in self.testlist if isSpam.match(spam)]
		#self.TestDataHams = [ham for ham in self.testlist if not isSpam.match(ham)]
		
		for f in self.testlist:
			classification = self.filecategorization(f)
			self.CategoryResults.append(classification)
		with open("nboutput.txt","w") as _filehandle:
			_filehandle.write("\n".join(self.CategoryResults))
			_filehandle.close()

Args_p = argparse.ArgumentParser(usage="python nbclassify.py <INPUTDIR>", description="Categorize email as spam or ham ")
Args_p.add_argument('idir', help="inputdir")
args = Args_p.parse_args()

objNBClassify = NaiveBayesClassify()
objNBClassify.readmodeldata("nbmodel.txt")
objNBClassify.readTestData(args.idir)

objNBClassify.categorizeData()

#HamDocuments = len([ham for ham in objNBClassify.TestDataHams if ham not in objNBClassify.PredictedHams])

#SpamDocuments = len([spam for spam in objNBClassify.TestDataSpams if spam not in objNBClassify.PredictedSpams])

#documents = objNBClassify.model["Documents"]
#Total_accuracy = (documents - (HamDocuments + SpamDocuments))/documents
#Spam_Pre = (len(objNBClassify.PredictedSpams) - HamDocuments)/len(objNBClassify.PredictedSpams)
#Ham_Pre = (len(objNBClassify.PredictedHams) - SpamDocuments)/len(objNBClassify.PredictedHams)
#Spam_Rec = (len(objNBClassify.PredictedSpams) - HamDocuments)/(len(objNBClassify.PredictedSpams) - HamDocuments + SpamDocuments)
#Ham_Rec = (len(objNBClassify.PredictedHams) - SpamDocuments)/(len(objNBClassify.PredictedHams) - SpamDocuments + HamDocuments)
#SPAM_F1 =  2 * Spam_Pre * Spam_Rec / (Spam_Pre+ Spam_Rec)
#HAM_F1 =  2 * Ham_Pre * Ham_Rec / (Ham_Pre+ Ham_Rec)

#print("\tPre\tRec\tF1 ")
#print("Spam  {0}\t{1}\t{2}".format(round(Spam_Pre,2),round(Spam_Rec,2), round(SPAM_F1,2)))
#print("Ham   {0}\t{1}\t{2}".format(round(Ham_Pre,2),round(Ham_Rec,2),round(HAM_F1,2)))
#print("Weighted Avg : {0}".format(round((Spam_Pre+Spam_Rec+SPAM_F1+Ham_Pre+Ham_Rec+HAM_F1)/6,2)))
#print("Total Accuracy : {0}".format(Total_accuracy))