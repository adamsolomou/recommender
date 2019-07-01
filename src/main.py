import numpy as np
import pandas as pd
import tensorflow as tf
from utils import root_mean_square_error
from reader import fetch_data
from sklearn.linear_model import LinearRegression


# models
from bias_sgd import BiasSGD
from iterative_svd import IterativeSVD
from autoencoder import Autoencoder
from embeddings import Embeddings


device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print('GPU device not found')
    print('Be sure you want to continue...')
else:
    print('Found GPU at: {}'.format(device_name))

dataloader = fetch_data(train_size=0.88)
number_of_users, number_of_movies = 10000, 1000

# Training
train_IDs, train_users, train_movies, train_ratings, A_train = dataloader['train']

# Validation
valid_IDs, valid_users, valid_movies, valid_ratings, A_valid = dataloader['valid']

# Testing
test_IDs, test_users, test_movies = dataloader['test']

known_train = ~np.isnan(A_train)
known_validation = ~np.isnan(A_valid)

# Fill-in missing entries
X_train = pd.DataFrame(A_train, copy=True)

movie_avg_ratings = X_train.mean(axis=0, skipna=True)

A_train_zeros = X_train.fillna(0.0).values
A_train_average = X_train.fillna(movie_avg_ratings, axis=0).values
known_train_int = known_train.astype(int)

# iterative SVD model
model = IterativeSVD(shrinkage=38)
model.fit(A_train_average, known_train, A_valid=A_valid, valid_mask=known_validation, verbose=True)

preds_iSVD_val = model.predict(valid_users, valid_movies)
preds_iSVD_test = model.predict(test_users, test_movies)

# biased SGD model
model = BiasSGD(number_of_users=10000, number_of_movies=1000,
                hidden_size=12, regularization_matrix=0.08, regularization_vector=0.04)

model.fit(train_users, train_movies, train_ratings,
          valid_users=valid_users, valid_movies=valid_movies, valid_ratings=valid_ratings,
          num_epochs=50, decay=1.5, lr=0.05, decay_every=5, verbose=True)

preds_bSGD_val = model.predict(valid_users, valid_movies)
preds_bSGD_test = model.predict(test_users, test_movies)

# autoencoder model
model = Autoencoder(number_of_users, number_of_movies)
model.fit(A_train_zeros, known_train_int, valid_users=valid_users, valid_movies=valid_movies,
             valid_ratings=valid_ratings)

preds = model.predict(A_train_zeros, test_users, test_movies)

preds_autoencoder_val = model.predict(A_train_zeros, valid_users, valid_movies)
preds_autoencoder_test = model.predict(A_train_zeros, test_users, test_movies)

# embeddings model
model = Embeddings(number_of_users, number_of_movies)
model.fit(train_users, train_movies, train_ratings,
          valid_users=valid_users, valid_movies=valid_movies,
          valid_ratings=valid_ratings, epochs=5)

preds_embeddings_val = model.predict(valid_users, valid_movies)
preds_embeddings_test = model.predict(test_users, test_movies)

# create regressor for combined predictions
valid_predictions = np.concatenate([preds_iSVD_val, preds_bSGD_val, preds_autoencoder_val, preds_autoencoder_val])
valid_predictions = valid_predictions.reshape(4, valid_users.shape[0]).T

test_predictions = np.concatenate([preds_iSVD_test, preds_bSGD_test, preds_autoencoder_test, preds_autoencoder_test])
test_predictions = test_predictions.reshape(4, test_users.shape[0]).T

regressor = LinearRegression(fit_intercept=True)
regressor.fit(valid_predictions, valid_ratings)

regressor_predictions = regressor.predict(test_predictions)

# persist results
preds_pd = pd.DataFrame(index=test_IDs, data=regressor_predictions, columns=['Prediction'])
preds_pd.to_csv('ensemble_predictions.csv')