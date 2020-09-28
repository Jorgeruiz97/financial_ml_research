import numpy as np
import pandas as pd
import random


class Simulation:
    def __init__(self, epochs=10000, returns_len=10000, og_assets=23, corr_assets=7, sigma1=.25, rep=True):
        self.name = 'montecarlo simulation'
        self.epochs = epochs
        self.returns_len = returns_len
        self.og_assets = og_assets
        self.corr_assets = corr_assets
        self.sigma1 = sigma1
        self.rep = rep
        self.results = pd.Dataframe([])

    # Time series of correlated variables
    def generate_data(self, returns_len=10000, og_assets=23, corr_assets=5, sigma1=.25, rep=True):
        """
        This function returns a pandas dataframe with random returns.

        This function generates [og_assets] columns of random returns and then
        picks [corr_assets] columns and duplicates those returns + a difference
        to have correlated columns.

        Parameters:
        row_len = number of rows
        og_assets = original number of assets
        corr_assets = number of assets to randomly select and generate
        sigma1 = standard deviation to randomly generate correlated assets

        returns:
        x = a pandas dataframe with the returns
        cols = an array with the cols name that were used to create correlated cols
        dependency = An array of tuples with the cols based on other cols
        """
        # set seed to replicate experiment
        if rep is True:
            np.random.seed(seed=12345)
            random.seed(12345)

        # Generate random returns between 0 and 1 using a normal distribution
        # nobs = number of rows
        # size0 = number of columns
        x = np.random.normal(0, 1, size=(returns_len, og_assets))

        # make an array from [0, ..., corr_assets]
        carange = range(corr_assets)

        # Pick random columns to duplicate and make extra correlated returns
        cols = [random.randint(0, og_assets - 1) for i in carange]

        # generate extra returns for the correlated columns
        z = np.random.normal(0, sigma1, size=(returns_len, len(cols)))

        # Create a new dataframe with the selected columns to replicate
        # Add the extra returns to make them different, but correlated.
        y = x[:, cols] + z

        # column bind new df with correlated returns with original df
        x = np.append(x, y, axis=1)

        # get the number of cols and adjust for zero [0,1,2] -> [1,2,3]
        column_names = x.shape[1]+1

        # transform newly created dataframe to pandas type
        x = pd.DataFrame(x, columns=range(1, column_names))

        # which columns are based on what other columns
        # Ex. [(3, 6), (4, 7)]... column 6 is based on column 3
        dependency = [(j+1, og_assets+i) for i, j in enumerate(cols, 1)]

        return x, cols, dependency

        def run(self, epochs=100000, models=[]):
            """
            rows will be epochs and columns will be results per model
            """
            for i in range(0, epochs):
                for model in models:
                    # execute model and calculate weights
                    # calculate performance metrics
                    # caculate risk metrics
                    # save to results df
                    pass
