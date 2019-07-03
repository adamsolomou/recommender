import argparse

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


def create_parser():
    parser = argparse.ArgumentParser(description="Run cross validation for model")
    parser.add_argument("--verbose", "-v", action="store_true")

    parser.add_argument("--splits-num", type=int, default=10)
    parser.add_argument("--shuffle", action="store_true")

    parser.add_argument("--hidden-size", type=int, default=12)
    parser.add_argument("--regularization-matrix", type=float, default=0.08)
    parser.add_argument("--regularization-vector", type=float, default=0.04)

    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--decay", type=float, default=1.5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--decay-every", type=int, default=5)

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    # shuffle to ensure train examples exist for all users with high
    # probability
    kf = KFold(n_splits=args.splits_num, shuffle=args.shuffle, random_state=42)

    predictions = np.zeros(users.shape[0])

    # shuffle to

    for train_index, valid_index in kf.split(users):
        train_users = users[train_index]
        train_movies = movies[train_index]
        train_ratings = ratings[train_index]

        valid_users = users[valid_index]
        valid_movies = movies[valid_index]
        valid_ratings = ratings[valid_index]

        model = BiasSGD(number_of_users=number_of_users,
                        number_of_movies=number_of_movies,
                        hidden_size=args.hidden_size,
                        regularization_matrix=args.regularization_matrix,
                        regularization_vector=args.regularization_vector)

        model.fit(train_users, train_movies, train_ratings,
                  valid_users=valid_users, valid_movies=valid_movies,
                  valid_ratings=valid_ratings,
                  num_epochs=args.num_epochs, decay=args.decay, lr=args.lr,
                  decay_every=args.decay_every,
                  verbose=args.verbose)

        predictions[valid_index] = model.predict(valid_users, valid_movies)

    # print final RMSE on all the results
    print(root_mean_square_error(ratings, predictions))
