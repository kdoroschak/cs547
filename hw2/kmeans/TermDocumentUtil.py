import scipy.io as sio
import numpy as np

class TermDocument:
	def __init__(self, filename):
		self.is_sparse = True
		self.is_tfidf = False
		self.matrix = self.read_matrix_market(filename)

	def read_matrix_market(self, filename):
		self.sparse = True
		return sio.mmread("bbc.mtx").T

	def convert_to_tfidf(self, tf_data):
		""" Convert a term-doc-frequency matrix into a term-doc-tfidf matrix.
		For each term, take the average tfidf over each class.
		"""
		tf_data = tf_data.toarray()
		tfidf = np.zeros_like(tf_data)
		N,d = tf_data.shape
		for doc_idx in range(N):
			doc = tf_data[doc_idx,:]
			for term in range(d):
				tfidf[doc_idx, term] = self.tfidf(term, doc, tf_data)
		self.matrix = tfidf
		self.is_tfidf = True
		self.is_sparse = False
		return tfidf

	def tf(self, t, d):
		freq_t_in_d = d[t]
		max_freq_in_d = np.max(d)
		tf = freq_t_in_d / float(max_freq_in_d)
		return tf

	def idf(self, t, D):
		N = D.shape[0]
		idf_weight = 1 + np.sum(D[:,t])		# Avoid divide by 0
		idf = np.log(float(N)/idf_weight)
		return idf

	def tfidf(self, t, d, D):
		tfidf = self.tf(t, d) * self.idf(t, D)
		return tfidf

	