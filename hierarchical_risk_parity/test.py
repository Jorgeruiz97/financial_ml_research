import numpy as np
import pandas as pd
import random


def generate_data(row_len, og_assets, corr_assets, sigma1, rep=True):
    """
    This method generates returns a pandas dataframe with random returns.

    This function generates [og_assets] columns of random returns and then
    picks [corr_assets] columns and duplicates those returns + a difference
    to have correlated columns.
    """
    # set seed to replicate experiment
    if rep is True:
        np.random.seed(seed=12345)
        random.seed(12345)

        # Generate random returns between 0 and 1 using a normal distribution
        # nobs = number of rows
        # size0 = number of columns
        x = np.random.normal(0, 1, size=(row_len, og_assets))

        # make an array from [0, ..., corr_assets]
        carange = range(corr_assets)

        # Pick random columns to duplicate and make extra correlated returns
        selected_columns = [random.randint(0, og_assets - 1) for i in carange]

        # generate extra returns for the correlated columns
        z = np.random.normal(0, sigma1, size=(row_len, len(selected_columns)))

        # Create a new dataframe with the selected columns to replicate
        # Add the extra returns to make them different, but correlated.
        y = x[:, selected_columns] + z

        # column bind new df with correlated returns with original df
        x = np.append(x, y, axis=1)

        # get the number of cols and adjust for zero [0,1,2] -> [1,2,3]
        column_names = x.shape[1]+1

        # transform newly created dataframe to pandas type
        x = pd.DataFrame(x, columns=range(1, column_names))

        return x, selected_columns

# This is a proper distance metric


def correl_dist(corr):
    dist = ((1-corr)/2.)**.5  # distance matrix
    return dist


x, cols = generate_data(10000, 5, 5, .25)

corr = x.corr()

dist = correl_dist(corr)

print(corr)
print('')
print(dist)
