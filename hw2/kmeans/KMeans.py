import numpy as np
from matplotlib import pyplot as plt
import random
from scipy.spatial import distance

class KMeans:
	def __init__(self, k, Z, x):
		pass
		self.k = k
		self.mu = self.random_init_centers(k, x) #cluster centers , size kx2
		self.Z = Z # cluster assignments (len(Z) == len(x))
		self.x = x # data points -- list of tuples

	def random_init_centers(self, k, x):
		# mu = [[random.random(), random.random()] for _ in xrange(k)]
		mu = []
		for i in range(k):
			mu.append(random.choice(x))
		return mu

	def identify_clusters(self, mu, x):
		Z = np.copy(self.Z)
		changed = False
		for i,point in enumerate(x):
			distances = distance.cdist(mu, [point], 'euclidean')
			newCluster = np.argmin(distances)
			if not changed and Z[i] != newCluster:
				changed = True
			Z[i] = newCluster
		return Z, changed

	def calculate_centers(self, Z, x):
		mu = np.zeros((self.k, 2))
		cluster_count = [0 for _ in xrange(self.k)]
		for i,z in enumerate(Z):
			mu[z]+=x[i]
			cluster_count[z]+=1
		print mu
		print cluster_count
		print mu[:,0]
		mu[:,0] = np.divide(mu[:,0],cluster_count)
		mu[:,1] = np.divide(mu[:,1],cluster_count)
		print mu
		return mu

	def run_kmeans(self):
		converged = False
		iteration = 0
		while not converged:
			iteration += 1
			print "iteration:", iteration
			# E step
			self.Z, changed = self.identify_clusters(self.mu, self.x)
			print changed

			# M step
			self.mu = self.calculate_centers(self.Z, self.x)

			if changed == False:
				converged=True

		self.plot_clustering(self.mu, self.Z, self.x)

		return

	

	def plot_clustering(self, mu, Z, x, savefile="plot.png"):
		print "plotting...."
		clusters = [[] for _ in xrange(self.k)]
		# this will run out of colors for k>21 #from tableau20 + black so far
		colors = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765), (0.6823529411764706, 0.7803921568627451, 0.9098039215686274), (1.0, 0.4980392156862745, 0.054901960784313725), (1.0, 0.7333333333333333, 0.47058823529411764), (0.17254901960784313, 0.6274509803921569, 0.17254901960784313), (0.596078431372549, 0.8745098039215686, 0.5411764705882353), (0.8392156862745098, 0.15294117647058825, 0.1568627450980392), (1.0, 0.596078431372549, 0.5882352941176471), (0.5803921568627451, 0.403921568627451, 0.7411764705882353), (0.7725490196078432, 0.6901960784313725, 0.8352941176470589), (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), (0.7686274509803922, 0.611764705882353, 0.5803921568627451), (0.8901960784313725, 0.4666666666666667, 0.7607843137254902), (0.9686274509803922, 0.7137254901960784, 0.8235294117647058), (0.4980392156862745, 0.4980392156862745, 0.4980392156862745), (0.7803921568627451, 0.7803921568627451, 0.7803921568627451), (0.7372549019607844, 0.7411764705882353, 0.13333333333333333), (0.8588235294117647, 0.8588235294117647, 0.5529411764705883), (0.09019607843137255, 0.7450980392156863, 0.8117647058823529), (0.6196078431372549, 0.8549019607843137, 0.8980392156862745), (0,0,0)]
		for i,z in enumerate(Z):
			clusters[z].append(x[i])
		
		fig, ax = plt.subplots()

		for k in range(self.k):
			# fix plotting
			print np.array(clusters[k]).T[0]
			plt.scatter(np.array(clusters[k]).T[0], np.array(clusters[k]).T[1], figure=fig, color=random.choice(colors))
			plt.scatter(mu[k][0], mu[k][1], figure=fig, color='black', marker="+")

		fig.savefig(savefile, format="png")
		return

def main():
	data = np.loadtxt("2DGaussianMixture.csv", skiprows=1, delimiter=",")
	# print np.shape(data)
	# print data[0]
	labels = data[:,0]
	labels = [int(x) - int(min(labels)) for x in labels] # subtract the min so we have 0-based indexing
	points = data[:,1:]
	kmeans = KMeans(3, labels, points)
	kmeans.plot_clustering([[0,0], [0,0], [0,0]], labels, points)
	kmeans.run_kmeans()
	pass

if __name__ == "__main__":
	main()



