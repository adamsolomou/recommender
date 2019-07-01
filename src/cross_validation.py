import numpy as np
import tensorflow as tf
from utils import root_mean_square_error

# models
from bias_sgd import BiasSGD
from iterative_svd import IterativeSVD
from autoencoder import Autoencoder
from embeddings import Embeddings
from sklearn.model_selection import KFold
from reader import fetch_data

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print('GPU device not found')
    print('Be sure you want to continue...')
else:
    print('Found GPU at: {}'.format(device_name))


"""
Train on training split, predictions on validation split. 
"""

dataloader = fetch_data(train_size=1)
number_of_users, number_of_movies = 10000, 1000

IDs, users, movies, ratings, _ = dataloader['train']



# shuffle to ensure train examples exist for all users with high probability
kf = KFold(n_splits=10, shuffle=True, random_state=42)

predictions = np.zeros(users.shape[0])

# shuffle to

for train_index, valid_index in kf.split(users):
    train_users = users[train_index]
    train_movies = movies[train_index]
    train_ratings = ratings[train_index]

    valid_users = users[valid_index]
    valid_movies = movies[valid_index]
    valid_ratings = ratings[valid_index]

    model = BiasSGD(number_of_users=number_of_users, number_of_movies=number_of_movies,
                    hidden_size=12, regularization_matrix=0.08, regularization_vector=0.04)

    model.fit(train_users, train_movies, train_ratings,
              valid_users=valid_users, valid_movies=valid_movies, valid_ratings=valid_ratings,
              num_epochs=1, decay=1.5, lr=0.05, decay_every=5, verbose=True)

    predictions[valid_index] = model.predict(valid_users, valid_movies)

# print final RMSE on all the results
print(root_mean_square_error(ratings, predictions))
