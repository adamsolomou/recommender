import os
import numpy as np
import pandas as pd
import tensorflow as tf

# Custom dependencies 
from svdplus import SVDplus
from embeddings import Embeddings
from autoencoder import Autoencoder
from iterative_svd import IterativeSVD

from reader import fetch_data
from utils import root_mean_square_error

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')

print('Found GPU at: {}'.format(device_name))

n_users = 10000
n_movies = 1000

dataloader = fetch_data(train_size=0.88)

# Training 
train_users, train_movies, train_ratings, A_train = dataloader['train']

# Validation 
valid_users, valid_movies, valid_ratings, A_valid = dataloader['valid']

# Testing 
test_users, test_movies = dataloader['test']

unknown_train = np.isnan(A_train)
known_train = ~unknown_train

# Fill-in missing entries 
X_train = pd.DataFrame(A_train, copy=True)

movie_avg_ratings = X_train.mean(axis=0, skipna=True)

A_train_zeros = X_train.fillna(0.0).values
A_train_average = X_train.fillna(movie_avg_ratings, axis=0).values
known_train_int = known_train.astype(int)

###
model1 = IterativeSVD(n_users, n_movies)
model1.fit_data(A_train_average, train_users, train_movies, train_ratings)

preds = model1.predict(valid_users, valid_movies)
print(root_mean_square_error(valid_ratings, preds))
###

###
model2 = SVDplus(n_users, n_movies)
model2.fit_data(train_users, train_movies, train_ratings,
                users_validation=valid_users, movies_validation=valid_movies,
                ratings_validation=valid_ratings)

preds = model2.predict(valid_users, valid_movies)
print(root_mean_square_error(valid_ratings, preds))
###

###
model3 = Autoencoder(n_users, n_movies)
model3.train(A_train_zeros, known_train_int, users_validation=valid_users, movies_validation=valid_movies,
             ratings_validations=valid_ratings)

preds = model3.predict(A_train_zeros, valid_users, valid_movies)
print(root_mean_square_error(valid_ratings, preds))
###

###
model4 = Embeddings(n_users, n_movies)
model4.fit_data(train_users, train_movies, train_ratings,
                users_validation=valid_users, movies_validation=valid_movies,
                ratings_validation=valid_ratings)

preds = model4.predict(valid_users, valid_movies)
root_mean_square_error(valid_ratings, preds)
###