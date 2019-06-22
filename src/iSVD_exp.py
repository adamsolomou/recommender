import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm 
from sklearn.model_selection import KFold, StratifiedKFold

from iterative_svd import IterativeSVD
from utils import root_mean_square_error

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

n_users, n_movies = (10000, 1000)

data = pd.read_csv('data/data_train.csv')

# Configurations to test 
number_of_iterations = 21

sing_values_to_keep = list()
sing_values_to_keep.append([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 6, 10])
sing_values_to_keep.append(np.arange(2,23).tolist())
sing_values_to_keep.append(2*np.ones(number_of_iterations, dtype=int).tolist())
sing_values_to_keep.append(12*np.ones(number_of_iterations, dtype=int).tolist())
sing_values_to_keep.append([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 6, 12])
sing_values_to_keep.append([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 12])
sing_values_to_keep.append([2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12])

kFold = KFold(n_splits=10)

cv_error_per_iter = list()
number_of_models_to_test = len(sing_values_to_keep)

# Repeat for each model configuration 
for model_config in tqdm(sing_values_to_keep): 
    # Initialize array 
    val_error_per_iter = list()

    # Cross-Validation 
    for train_msk, valid_msk in kFold.split(data):
        train = data.iloc[train_msk]
        valid = data.iloc[valid_msk]

        # Training data
        users_train = list()
        movies_train = list()
        ratings_train = train['Prediction'].values

        A_train = np.zeros(shape=(n_users, n_movies))
        A_train[:] = np.nan

        for i, (user, movie) in enumerate(train.Id.str.extract('r(\d+)_c(\d+)').values):
            # Cast 
            u_idx = int(user) - 1
            m_idx = int(movie) - 1

            users_train.append(u_idx)
            movies_train.append(m_idx)

            A_train[u_idx, m_idx] = ratings_train[i]

        users_train = np.array(users_train)
        movies_train = np.array(movies_train)

        # Validation data 
        users_valid = list()
        movies_valid = list()
        ratings_valid = valid['Prediction'].values

        A_valid = np.zeros(shape=(n_users, n_movies))
        A_valid[:] = np.nan

        for i, (user, movie) in enumerate(valid.Id.str.extract('r(\d+)_c(\d+)').values):
            # Cast 
            u_idx = int(user) - 1
            m_idx = int(movie) - 1

            users_valid.append(u_idx)
            movies_valid.append(m_idx)

            A_valid[u_idx, m_idx] = ratings_valid[i]

        users_valid = np.array(users_valid)
        movies_valid = np.array(movies_valid)


        #######################################

        # Fill-in missing entries 
        X_train = pd.DataFrame(A_train, copy=True)

        movie_avg_ratings = X_train.mean(axis=0, skipna=True)

        A_train_average = X_train.fillna(movie_avg_ratings, axis=0).values

        # Initialize model
        model = IterativeSVD(n_users, n_movies, model_config)
        error_per_iter = model.fit_data(A_train_average, 
                                        users_train, 
                                        movies_train, 
                                        ratings_train, 
                                        users_valid, 
                                        movies_valid, 
                                        ratings_valid)

        val_error_per_iter.append(error_per_iter)

    cv_error_per_iter.append(np.mean(val_error_per_iter, axis=0))

np.save('logs/iSVD/cv_error_per_iter.npy', cv_error_per_iter)
np.save('cv_error_per_iter.npy', cv_error_per_iter)

plt.figure()
model_id = 1
for result in cv_error_per_iter:
    plt.plot(np.arange(1,22), result, '--', label='Config {}'.format(model_id))
    model_id += 1
plt.ylabel('Cross-Validation RMSE')
plt.xlabel('Iteration')
plt.legend()
plt.grid(True)
plt.savefig('cv_error_per_iter.eps', format='eps', dpi=1000)

