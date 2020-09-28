import matplotlib.pyplot as mpl
import scipy.cluster.hierarchy as sch
import numpy as np
import pandas as pd
from pprint import pprint


class HRP:
    def __init__(self, returns, cluster_type='single'):
        self.name = 'hierarchical Risk Parity'
        self.returns_cov = returns.cov()
        self.returns_corr = returns.corr()
        self.returns_corr_dist = self.correl_dist(self.returns_corr)
        self.link = sch.linkage(self.returns_corr_dist, cluster_type)
        self.weights = self.calculate_weights()

    # Compute the inverse-variance portfolio
    def get_ivp(self, cov, **kargs):
        ivp = 1./np.diag(cov)
        ivp /= ivp.sum()
        return ivp

    # Compute variance per cluster
    def get_cluster_var(self, cov, c_items):
        cov_ = cov.loc[c_items, c_items]  # matrix slice
        w_ = self.get_ivp(cov_).reshape(-1, 1)
        c_var = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
        return c_var

    # Sort clustered items by distance
    def get_quasi_diag(self, link):
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]  # number of original items
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0]*2, 2)  # make space
            df0 = sort_ix[sort_ix >= num_items]  # find clusters
            i = df0.index
            j = df0.values-num_items
            sort_ix[i] = link[j, 0]  # item 1

            df0 = pd.Series(link[j, 1], index=i+1)
            sort_ix = sort_ix.append(df0)  # item 2
            sort_ix = sort_ix.sort_index()  # re-sort
            sort_ix.index = range(sort_ix.shape[0])  # re-index
        return sort_ix.tolist()

    # Compute HRP alloc
    def get_rec_bipart(self, cov, sort_ix):
        w = pd.Series(1, index=sort_ix)
        c_items = [sort_ix]  # initialize all items in one cluster

        print('cItems: \n')
        while len(c_items) > 0:
            pprint(c_items)

            # bi-section
            c_items = [i[j:k] for i in c_items for j, k in ((0, int(len(i)/2)), (int(len(i)/2), len(i))) if len(i) > 1]
            for i in range(0, len(c_items), 2):  # parse in pairs
                c_items0 = c_items[i]  # cluster 1
                c_items1 = c_items[i+1]  # cluster 2
                c_var0 = self.get_cluster_var(cov, c_items0)
                c_var1 = self.get_cluster_var(cov, c_items1)
                alpha = 1-c_var0/(c_var0+c_var1)
                w[c_items0] *= alpha  # weight 1
                w[c_items1] *= 1-alpha  # weight 2
        return w

    # A distance matrix based on correlation, where 0<=d[i,j]<=1
    # This is a proper distance metric
    def correl_dist(self, corr):
        # dist = ((1-corr)/2.)**.5  # distance matrix
        dist = ((1-self.returns_corr)/2.)**.5  # distance matrix
        return dist

    # Heatmap of the correlation matrix
    def plot_corr_matrix(self, path, labels=None):
        c = self.sorted_corr
        if labels is None:
            labels = []
        mpl.pcolor(c)
        mpl.colorbar()
        mpl.yticks(np.arange(.5, c.shape[0]+.5), labels)
        mpl.xticks(np.arange(.5, c.shape[0]+.5), labels)
        mpl.savefig(path)
        mpl.clf()
        mpl.close()  # reset pylab

    def plot_dendogram(self, path):
        sch.dendrogram(self.link)
        mpl.savefig(path)
        mpl.clf()
        mpl.close()  # reset pylab

    def calculate_weights(self):
        sort_ix = self.get_quasi_diag(self.link)
        sort_ix = self.returns_corr.index[sort_ix].tolist()  # recover labels

        # ordered correlation matrix
        self.sorted_corr = self.returns_corr.loc[sort_ix, sort_ix]  # reorder

        # Assign weight to each asset
        capital_alloc = self.get_rec_bipart(self.returns_cov, sort_ix)
        return capital_alloc
