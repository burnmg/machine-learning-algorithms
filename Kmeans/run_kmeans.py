import numpy as np
from kmeans import kmeans
import matplotlib.pyplot as plt

x = np.vstack(((np.random.randn(150, 2) * 0.75 + np.array([5, 0])),
                  (np.random.randn(50, 2) * 0.25 + np.array([-0.5, 0.5])),
                  (np.random.randn(50, 2) * 0.5 + np.array([-0.5, -0.5]))))


centroids, cluster_assignments = kmeans(x, 3)

plt.scatter(x[:,0], x[:,1], c=cluster_assignments)
plt.scatter(centroids[:, 0], centroids[:, 1], c="r", s=100)
plt.show()

