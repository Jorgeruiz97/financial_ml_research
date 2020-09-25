import matplotlib.pyplot as mpl
import scipy.cluster.hierarchy as sch
import random
import numpy as np
import pandas as pd


class HRP:
    def __init__(self):
        self.name = "hierarchical Risk Parity"
        print(self.name)

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
        while len(c_items) > 0:
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i)/2),
                (len(i)/2, len(i))) if len(i) > 1]  # bi-section
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
        dist = ((1-corr)/2.)**.5  # distance matrix
        return dist

    # Heatmap of the correlation matrix
    def plot_corr_matrix(self, path, corr, labels=None):
        if labels is None:
            labels = []
        mpl.pcolor(corr)
        mpl.colorbar()
        mpl.yticks(np.arange(.5, corr.shape[0]+.5), labels)
        mpl.xticks(np.arange(.5, corr.shape[0]+.5), labels)
        mpl.savefig(path)
        mpl.clf()
        mpl.close()  # reset pylab
        return

    # Time series of correlated variables
    def generate_data(self, nobs, size0, size1, sigma1):
        # 1) generating some uncorrelated data
        np.random.seed(seed=12345)
        random.seed(12345)
        # each row is a variable
        x = np.random.normal(0, 1, size=(nobs, size0))
        # 2) creating correlation between the variables
        cols = [random.randint(0, size0 - 1) for i in range(size1)]
        y = x[:, cols]+np.random.normal(0, sigma1, size=(nobs, len(cols)))
        x = np.append(x, y, axis=1)
        x = pd.DataFrame(x, columns=range(1, x.shape[1]+1))
        return x, cols

    def main(self):
        # 1) Generate correlated data
        nobs, size0, size1, sigma1 = 10000, 5, 5, .25
        x, cols = self.generate_data(nobs, size0, size1, sigma1)

        dependency = [(j+1, size0+i) for i, j in enumerate(cols, 1)]
        print(dependency)

        cov, corr = x.cov(), x.corr()
        # 2) compute and plot correl matrix
        self.plot_corr_matrix('HRP3_corr0.png', corr, labels=corr.columns)
        # 3) cluster
        dist = self.correl_dist(corr)
        link = sch.linkage(dist, 'single')
        sort_ix = self.get_quasi_diag(link)
        sort_ix = corr.index[sort_ix].tolist()  # recover labels
        df0 = corr.loc[sort_ix, sort_ix]  # reorder
        # 4) Capital allocation
        self.plot_corr_matrix('HRP3_corr1.png', df0, labels=df0.columns)
        hrp = self.get_rec_bipart(cov, sort_ix)
        print(hrp)
        return
