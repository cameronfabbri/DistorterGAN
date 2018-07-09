# clustering dataset
# determine k using elbow method

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.misc as misc

x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
x2 = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])

paths  = glob.glob('../dataset/distorted/*.*')
images = []

for p in paths:
   images.append(misc.imresize(misc.imread(p), (256,256,3)).flatten())
X = np.asarray(images)

distortions = []
K = range(1,len(images))
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    print 'fitting',k
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
