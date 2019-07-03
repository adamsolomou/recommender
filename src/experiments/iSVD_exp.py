import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from reader import fetch_data
from iterative_svd import IterativeSVD
from utils import root_mean_square_error

dataloader = fetch_data(train_size=0.9)

# Training
train_IDs, train_users, train_movies, train_ratings, A_train = dataloader['train']

# Validation
valid_IDs, valid_users, valid_movies, valid_ratings, A_valid = dataloader['valid']

if __name__ == "__main__":

    val_error = list()
    shrinkage_array = np.arange(20, 51)

    known_train = ~np.isnan(A_train)
    known_validation = ~np.isnan(A_valid)

    # Fill-in missing entries
    X_train = pd.DataFrame(A_train, copy=True)

    movie_avg_ratings = X_train.mean(axis=0, skipna=True)

    A_train_average = X_train.fillna(movie_avg_ratings, axis=0).values

    for shrinkage in shrinkage_array: 

        model = IterativeSVD(shrinkage=shrinkage)

        model.fit(A_train_average, known_train, verbose=False)

        preds = model.predict(valid_users, valid_movies)

        val_error.append(root_mean_square_error(valid_ratings, preds))

    val_error = np.array(val_error)

    plt.figure()
    plt.plot(shrinkage_array, val_error)
    plt.ylabel('RMSE')
    plt.xlabel('Threshold')
    plt.show()
