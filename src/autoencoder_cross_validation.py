import numpy as np
import pandas as pd
import os
import tensorflow as tf
from utils import root_mean_square_error
from autoencoder import Autoencoder
from reader import fetch_data
from sklearn.model_selection import KFold
import argparse


device_name = tf.test.gpu_device_name()

print('Found GPU at: {}'.format(device_name))

DATA_DIR = '/cluster/home/saioanni/CIL/recommender/data/'

number_of_users, number_of_movies = (10000, 1000)
testing=True

dataloader = fetch_data(train_size=1)
number_of_users, number_of_movies = 10000, 1000
IDs, users, movies, ratings, _ = dataloader['train']


def create_parser():
    parser = argparse.ArgumentParser(description="Run cross validation for model")
    parser.add_argument("--verbose", "-v", action="store_true")

    parser.add_argument("--splits-num", type=int, default=10)
    parser.add_argument("--shuffle", action="store_true")

    parser.add_argument("--hidden-layers", nargs='+', type=int, default=[15, 15, 15])
    parser.add_argument("--masking", type=float, default=0.5)

    parser.add_argument("--num-epochs", type=int, default=200)

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    kf = KFold(n_splits=args.splits_num, shuffle=args.shuffle, random_state=42)

    for fold, (train_index, valid_index) in enumerate(kf.split(users)):
        # Initialize matrices
        train_users = users[train_index]
        train_movies = movies[train_index]
        train_ratings = ratings[train_index]

        valid_users = users[valid_index]
        valid_movies = movies[valid_index]
        valid_ratings = ratings[valid_index]

        data_zeros = np.full((number_of_users, number_of_movies), 0)
        data_mask = np.full((number_of_users, number_of_movies), 0)

        for i, (user, movie) in enumerate(zip(train_users, train_movies)):
            data_zeros[user][movie] = train_ratings[i]
            data_mask[user][movie] = 1

        model3 = Autoencoder(number_of_users, number_of_movies, layers=args.hidden_layers, masking=args.masking)
        model3.train(data_zeros, data_mask, users_validation=valid_users, movies_validation=valid_movies,
                     ratings_validations=valid_ratings, n_epochs=args.num_epochs, verbose=False)

        preds = model3.predict(data_zeros, valid_users, valid_movies)

        score = root_mean_square_error(valid_ratings, preds)
        pint("fold:", fold, "score:", score)
