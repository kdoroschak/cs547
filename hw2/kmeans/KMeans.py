import numpy as np
import random
import scipy.io as sio
from scipy.spatial import distance
from matplotlib import pyplot as plt
from TermDocumentUtil import TermDocument
from ErrorUtil import ErrorModel

class KMeans:
	def __init__(self, k, Z, x, init_method="random_init_centers", centers=None):
		pass
		self.k = k
		self.Z_actual = Z
		self.Z = np.zeros_like(Z) # cluster assignments (len(Z) == len(x))	
		self.x = x # data points -- list of tuples
		if init_method == "random_init_centers":
			self.mu = self.random_init_centers(k, x)
		elif init_method == "kmeans_plus_plus_init":
			self.mu = self.kmeans_plus_plus_init(k, x)
		elif init_method == "given":
			if centers is None:
				raise TypeError("Must specify centers.")
			self.mu = centers 	# Should be a list of numpy arrays
		else:
			raise NameError(init_method + " is not defined.")

	def random_init_centers(self, k, x):
		mu = []
		for i in range(k):
			mu.append(random.choice(x))
		return mu

	def kmeans_plus_plus_init(self, k, x):
		mu = []
		# (i) Choose mu_0 uniformly at random
		mu.append(random.choice(x))
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
			mu.append(mu_new)
		return mu

	def identify_clusters(self, mu, x):
		Z = np.copy(self.Z)
		assignments_changed = False
		for i,point in enumerate(x):
			distances = distance.cdist(mu, [point], 'euclidean')
			new_cluster = np.argmin(distances)
			if not assignments_changed and Z[i] != new_cluster:
				assignments_changed = True
			Z[i] = new_cluster
		return Z, assignments_changed

	def calculate_centers(self, Z, x):
		mu = np.zeros((self.k, x.shape[1]))
		cluster_count = [1 for _ in xrange(self.k)]
		for i,z in enumerate(Z):
			mu[z]+=x[i]
			cluster_count[z]+=1

		mu[:,0] = np.divide(mu[:,0],cluster_count)
		mu[:,1] = np.divide(mu[:,1],cluster_count)
		return mu

	def run_kmeans(self,max_iter=150):
		converged = False
		iteration = 0
		errors = []
		while not converged and iteration < max_iter:
			iteration += 1
			print "iteration:", iteration
			# E step
			self.Z, assignments_changed = self.identify_clusters(self.mu, self.x)

			# M step
			self.mu = self.calculate_centers(self.Z, self.x)

			if assignments_changed == False:
				converged=True

			err = ErrorModel(self.Z, self.Z_actual).zero_one_loss()
			errors.append(err) 
		return errors

	def plot_clustering_2D(self, savefile="plot.png"):
		self.plot_clustering_2D_(self.mu, self.Z, self.x, savefile=savefile)
		return

	def plot_clustering_2D_(self, mu, Z, x, savefile="plot.png"):
		print "plotting clusters..."
		clusters = [[] for _ in xrange(self.k)]
		# this will run out of colors for k>21 #from tableau20 + black so far
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
		for i,z in enumerate(Z):
			clusters[z].append(x[i])
		fig, ax = plt.subplots()

		for k in range(self.k):
			x1 = np.array(clusters[k]).T[0]
			x2 = np.array(clusters[k]).T[1]
			if x1.size > 0 and x2.size > 0: # Can happen for large K
				plt.scatter(x1, x2, figure=fig, color=colors[k])#random.choice(colors))
				plt.scatter(mu[k][0], mu[k][1], figure=fig, color='black', marker="+")
		fig.savefig(savefile, format="png")
		plt.close()
		return

	def plot_centers(self, mu, x, savefile="centers.png"):
		print "plotting centers..."
		fig, ax = plt.subplots()
		x1 = [xx[0] for xx in x]
		x2 = [xx[1] for xx in x]
		mu1 = [m[0] for m in mu]
		mu2 = [m[1] for m in mu]
		plt.scatter(x1, x2, figure=fig, color='gray')
		plt.scatter(mu1, mu2, figure=fig, color='black', marker="+")
		fig.savefig(savefile, format="png")
		plt.close()
		return

	def within_cluster_ss(self, mu, Z, x):
		clusters = [[] for _ in xrange(self.k)]
		ss = np.zeros(self.k)
		for i,z in enumerate(Z):
			clusters[z].append(x[i])

		for k in range(self.k):
			ss[k] = np.sum(np.square(mu[k] - clusters[k]))
		return ss

def main():
	run_section = "bbc"

	if run_section == "partb":
		# Load data
		data = np.loadtxt("2DGaussianMixture.csv", skiprows=1, delimiter=",")
		labels = data[:,0]
		# Subtract the minimum label value so we have 0-based indexing
		labels = [int(x) - int(min(labels)) for x in labels] 
		points = data[:,1:]

		# Run kmeans
		ks = [2, 3, 5, 10, 15, 20]
		for k in ks:
			kmeans = KMeans(k, labels, points)
			kmeans.run_kmeans(max_iter=25)
			kmeans.plot_clustering_2D(savefile="random_init_k"+str(k)+".png")

	elif run_section == "partc_d":
		# Load data
		data = np.loadtxt("2DGaussianMixture.csv", skiprows=1, delimiter=",")
		labels = data[:,0]
		# Subtract the minimum label value so we have 0-based indexing
		labels = [int(x) - int(min(labels)) for x in labels] 
		points = data[:,1:]

		# Run kmeans 20x
		number_of_runs = 20
		ss = np.zeros((number_of_runs,3))
		centers = []
		for run_i in range(number_of_runs):
			kmeans = KMeans(3, labels, points)#, "kmeans_plus_plus_init")
			kmeans.run_kmeans(max_iter=25)
			centers += list(kmeans.mu)
			ss[run_i,:] = kmeans.within_cluster_ss(kmeans.mu, kmeans.Z, kmeans.x)
			kmeans.plot_clustering_2D(savefile="partc_run"+str(run_i)+".png")
		kmeans.plot_centers(centers, kmeans.x)

		print "Minimum:", np.min(ss)
		print "Maximum:", np.max(ss)
		print "Mean:", np.mean(ss)
		print "SD:", np.std(ss)

	elif run_section == "bbc":
		# Load data
		td = TermDocument("bbc.mtx")
		data = td.convert_to_tfidf(td.matrix)
		centers = np.loadtxt("bbc.centers")
		centers = [centers[i,:] for i in range(centers.shape[0])]
		classes = np.loadtxt("bbc.classes", dtype=int)[:,1]
		terms = np.loadtxt("bbc.terms", dtype=str)

		# Run kmeans
		kmeans = KMeans(len(centers), classes, data, "given", centers=centers)
		error = kmeans.run_kmeans(max_iter=5)

		# Plot classification error
		fig, ax = plt.subplots()
		plt.plot(range(len(error)), error)
		plt.xlabel("Number of iterations")
		plt.ylabel("Classification error (0/1 loss)")
		plt.savefig("bbc_kmeans_01loss.png")



if __name__ == "__main__":
	main()



