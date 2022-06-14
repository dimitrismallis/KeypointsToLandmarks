import faiss,math
import numpy as np
import time

class Kmeans(object):
    def __init__(self, k,dimension=256):
        self.k = k

        self.clus=faiss.Kmeans(dimension,self.k,niter=100,nredo=10,seed=1000,gpu=True)
        


    def cluster(self, data, verbose=False,featuredimension=256,centroids=None):
        end = time.time()

        xb=data

        # cluster the data
        I, index,centroids,distance= self.run_kmeans(xb, centroids,verbose)
        self.index=index
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return I,centroids.reshape(self.k ,featuredimension),distance

    def run_kmeans(self,x, centroids,verbose=False):

        self.clus.train(x, init_centroids=centroids)

        centroids = self.clus.centroids
        distance, I = self.clus.index.search(x, 1)

        loss = self.clus.obj[-1]

        if verbose:
            print('k-means loss : {0}'.format(loss))

        return [int(n[0]) for n in I], self.clus.index, centroids,distance


def preprocess_features(npdata):

    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata