import numpy as np

class ErrorModel:
	def __init__(self, predicted, actual):
		"""
		"""
		self.predicted = predicted
		self.actual = actual


	# ====== Binary classification error ======

	def compute_accuracy(self):
		TP = 0.
		TN = 0.
		FP = 0.
		FN = 0.
		for i in range(len(self.predicted)):
			if   self.predicted[i] == 1 and self.actual[i] == 1:
				TP += 1
			elif self.predicted[i] == 0 and self.actual[i] == 0:
				TN += 1
			elif self.predicted[i] == 1 and self.actual[i] == 0:
				FP += 1
			elif self.predicted[i] == 0 and self.actual[i] == 1:
				FN += 1	
		acc = (TP + TN) / (TP + TN + FP + FN)
		return acc

	def zero_one_loss(self):
		return