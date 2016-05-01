import numpy as np
import random
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
from matplotlib import pyplot as plt
from TermDocumentUtil import TermDocument
from ErrorUtil import ErrorModel

class EMGMM:
	def __init__(self, k, x, labels):
		self.k = k
		self.x = x
		self.actual_labels = labels
		self.predicted_labels = np.zeros_like(labels)
		self.w = np.zeros((x.shape[0], k), dtype=np.float64)# Can be 0, gets updated right away
		self.mu = self.init_mu_kmeans_plus_plus(k, x)		# Initialize as in kmeans++
		self.pi = np.ones(k, dtype=np.float64) / k 			# Initialize to uniform
		self.cov = self.init_cov_identity(k, x.shape[1])	# Initialize to identity
		self.predicted_clusters = None
		# pi -- mixture components (len(pi) == k)
		# data -- Nxd
		# w -- Nxk, w_{i,k} represents the membership of point i in cluster k
		#   sum(w,axis=0) == 1
		

	def init_mu_kmeans_plus_plus(self, k, x):
		mu = np.zeros((k, x.shape[1]),dtype=np.float64)
		# (i) Choose mu_0 uniformly at random
		mu[0,:] = random.choice(x)
		# (ii) Calculate distances from the last mu to all the data points
		for j in np.arange(1,k):
			min_dist = np.empty(x.shape[0])
			min_dist.fill(np.inf)
			for i,m in enumerate(mu):
				dist = np.sqrt(np.sum(np.square(x - m), axis=1)) # all the distances from 
				dist = [d.item() for d in dist]
				min_dist = np.minimum(min_dist, dist)			
			# Select the next mu from x with prob proportional to dist^2
			probs = np.square(min_dist)
			probs /= np.sum(probs)
			index = np.random.choice(range(x.shape[0]),p=probs)
			mu_new = x[index,:]
			mu[j,:] = mu_new
		return mu

	def init_cov_identity(self, k, d):
		# List of k dxd matrices
		cov = [np.identity(d) for i in range(k)]
		return cov

	def compute_weights(self, x, pi, mu, cov):
		# compute the "membership weight" of data point xi in cluster k, given parameters theta
		N,d = x.shape
		k = len(pi)
		w = np.zeros((N,k),np.float64)
		for i in range(N):
			for k in range(len(pi)):
				w[i,k] = pi[k] * self.mvn(x[i,:], mu[k], cov[k])
		sum_w = np.sum(w, axis=1)
		for k in range(len(pi)):
			w[:,k] /= sum_w
		assert w.shape == (N,len(pi))
		# TODO need to fix underflow :( :( :(
		return w

	def compute_means(self, w, x):
		mu = w.T.dot(x)
		w_sum = np.sum(w, axis=0)
		for d in range(mu.shape[1]):
			mu[:,d] /= w_sum
		assert len(mu) == np.shape(w)[1]
		return mu

	def compute_cov(self, w, x, mu):
		cov = []
		for k in range(w.shape[1]):
			cov_k = np.zeros((x.shape[1], x.shape[1]), np.float64) # dxd cov matrix
			for i in range(x.shape[0]):
				x_minus_mu = np.mat(x[i,:]-mu[k])
				debug = w[i,k] * x_minus_mu.T * x_minus_mu # should be dxd
				assert debug.shape == (x.shape[1], x.shape[1])
				cov_k += debug
			cov_k /= np.sum(w[:,k])
			cov.append(cov_k)
		assert len(cov) == w.shape[1]
		return np.array(cov)

	def compute_mixing_probabilities(self, w):
		pi = np.sum(w, axis=0) / np.shape(w)[0]
		assert len(pi) == np.shape(w)[1]
		return pi

	def mvn(self,x,mu,cov):
		var = multivariate_normal(mean=mu, cov=cov, allow_singular=True)
		return var.pdf(x)

	def gmm(self, max_iter=100):
		converged = False
		iteration = 0
		errors = []
		while not converged and iteration < max_iter:
			iteration += 1
			print "iteration:", iteration
			# E step
			self.w = self.compute_weights(self.x, self.pi, self.mu, self.cov)

			# M step
			self.mu = self.compute_means(self.w, self.x)
			self.cov = self.compute_cov(self.w, self.x, self.mu)
			self.pi = self.compute_mixing_probabilities(self.w)

			# TODO implement log-likelihood function and use it to determine convergence
			# if assignments_changed == False:
				# converged=True
			self.assign_clusters(self.w)
			err = ErrorModel(self.predicted_labels, self.actual_labels).zero_one_loss()
			errors.append(err) 

		return errors

	def assign_clusters(self, w):
		clusters = np.argmax(w, axis=1)
		self.predicted_clusters = clusters
		return clusters

	def plot_clustering_2D(self, savefile="plot.png"):
		print "plotting clusters..."
		clusters = [[] for _ in xrange(self.k)]
		# Tableau colors! Need to make a plotting helper someday to not have to do this...
		colors = [(0.12156, 0.46666, 0.70588),
				 (1.00000, 0.49803, 0.05490), 
				 (0.17254, 0.62745, 0.17254), 
				 (0.83921, 0.15294, 0.15686),
				 (0.58039, 0.40392, 0.74117), 
				 (0.54901, 0.33725, 0.29411), 
				 (0.89019, 0.46666, 0.76078), 
				 (0.49803, 0.49803, 0.49803), 
				 (0.73725, 0.74117, 0.13333), 
				 (0.09019, 0.74509, 0.81176), 
				 (0.00000, 0.00000, 0.00000),
				 (0.68235, 0.78039, 0.90980),
				 (1.00000, 0.73333, 0.47058), 
				 (0.59607, 0.87450, 0.54117),
				 (1.00000, 0.59607, 0.58823),
				 (0.77254, 0.69019, 0.83529),
				 (0.76862, 0.61176, 0.58039),
				 (0.96862, 0.71372, 0.82352), 
				 (0.78039, 0.78039, 0.78039),
				 (0.85882, 0.85882, 0.55294), 
				 (0.61960, 0.85490, 0.89803)]
		
		# Separate out the points in x by the cluster assignments in Z
		for i,c in enumerate(self.predicted_clusters):
			clusters[c].append(self.x[i])
		fig, ax = plt.subplots()

		for k in range(self.k):
			x1 = np.array(clusters[k]).T[0]
			x2 = np.array(clusters[k]).T[1]
			if x1.size > 0 and x2.size > 0: # Can happen for large K
				plt.scatter(x1, x2, figure=fig, color=colors[k])#random.choice(colors))
				center = (self.mu[k,0], self.mu[k,1])
				eigvals, eigvecs = np.linalg.eigh(self.cov[k])
        		eigvecs[eigvals.argsort()[::-1]]
        		eigvals[eigvals.argsort()[::-1]]
        		angle = np.degrees(np.arctan2(*eigvecs[:,0][::-1]))
        		width, height = 2 * 1.5 * np.sqrt(eigvals) # capture points within 1.5 st. dev.
        		e = Ellipse(xy=center, width=width, height=height, angle=angle)
        		e.set_alpha(95)
    			e.set_facecolor(colors[k])
        		ax.add_artist(e)
		fig.savefig(savefile, format="png")
		plt.close()
		return


def main():
	run_section = "bbc"

	if run_section == "2d_gaussian":
		# Load data
		data = np.loadtxt("2DGaussianMixture.csv", skiprows=1, delimiter=",")
		labels = data[:,0]
		# Subtract the minimum label value so we have 0-based indexing
		labels = [int(x) - int(min(labels)) for x in labels] 
		points = data[:,1:]

		k = 6
		em = EMGMM(k, points, labels)
		em.gmm(max_iter=100)
		labels = em.assign_clusters(em.w)
		em.plot_clustering_2D()

	if run_section == "bbc":
		# Load data
		td = TermDocument("bbc.mtx")
		data = td.convert_to_tfidf(td.matrix)
		centers = np.loadtxt("bbc.centers")
		centers = [centers[i,:] for i in range(centers.shape[0])]
		classes = np.loadtxt("bbc.classes", dtype=int)[:,1]
		terms = np.loadtxt("bbc.terms", dtype=str)

		# Run kmeans
		em = EMGMM(len(centers), data, classes)
		error = em.gmm(max_iter=3)

		# Plot classification error
		fig, ax = plt.subplots()
		plt.plot(range(len(error)), error)
		plt.xlabel("Number of iterations")
		plt.ylabel("Classification error (0/1 loss)")
		plt.savefig("bbc_gmm_01loss.png")


if __name__ == '__main__':
	main()