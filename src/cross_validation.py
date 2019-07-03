import argparse

import numpy as np
import pandas as pd
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

dataloader = fetch_data(train_size=1)
number_of_users, number_of_movies = 10000, 1000

IDs, users, movies, ratings, _ = dataloader['train']

def create_parser():
    parser = argparse.ArgumentParser(description="Run cross validation for model")
    parser.add_argument("--verbose", "-v", action="store_true")

    parser.add_argument("--splits-num", type=int, default=10)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--model", "-m", type=str)

    #BSGD parameters
    parser.add_argument("--hidden-size", type=int, default=9)
    parser.add_argument("--regularization-matrix", type=float, default=0.08)
    parser.add_argument("--regularization-vector", type=float, default=0.05)
    parser.add_argument("--decay", type=float, default=1.5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--decay-every", type=int, default=5)

    #Autoencoder parameters
    parser.add_argument("--hidden-layers", nargs='+', type=int, default=[15, 15, 15])
    parser.add_argument("--masking", type=float, default=0.5)
    parser.add_argument("--num-epochs", type=int, default=200)

    #Iterative SGD
    parser.add_argument("--shrinkage", type=int, default=38)
    parser.add_argument("--iterations", type=int, default=15)

    # Embeddings
    parser.add_argument("--embeddings-size", type=int, default=96)
    parser.add_argument("--embeddings-dropout-embeddings", type=float,default=0.2)

    parser.add_argument("--embeddings-dropout", type=float, default=0.1)
    parser.add_argument("--embeddings-decay", type=float, default=0.97)
    parser.add_argument("--embeddings-learning-rate", type=float, default=1.0)
    parser.add_argument("--embeddings-decay-steps", type=int, default=2000)
    parser.add_argument("--embeddings-num-epochs", type=int, default=5)
    parser.add_argument("--embeddings-batch-size", type=int, default=512)

    return parser

def iterative_svd(args):
    args.shuffle = True
    kf = KFold(n_splits=args.splits_num, shuffle=args.shuffle, random_state=42)

    score_lst = list()

    for fold, (train_index, valid_index) in enumerate(kf.split(users)):
        # Initialize matrices
        train_users = users[train_index]
        train_movies = movies[train_index]
        train_ratings = ratings[train_index]

        valid_users = users[valid_index]
        valid_movies = movies[valid_index]
        valid_ratings = ratings[valid_index]

        A_train = np.zeros(shape=(number_of_users, number_of_movies))
        A_train[:] = np.nan

        for i, (user, movie) in enumerate(zip(train_users, train_movies)):
            A_train[user][movie] = train_ratings[i]

        # mask 
        unknown_train = np.isnan(A_train)
        known_train = ~unknown_train

        # Fill-in missing entries 
        X_train = pd.DataFrame(A_train, copy=True)

        movie_avg_ratings = X_train.mean(axis=0, skipna=True)

        A_train_average = X_train.fillna(movie_avg_ratings, axis=0)
        A_train_average = A_train_average.values

        model = IterativeSVD(shrinkage=args.shrinkage)

        model.fit(A_train_average, known_train, iterations=args.iterations, verbose=False)

        preds = model.predict(valid_users, valid_movies)

        score = root_mean_square_error(valid_ratings, preds)
        score_lst.append(score)

        print("Fold:", fold+1, "RMSE:", score)

    print('Mean CV RMSE:', np.mean(score_lst))

def autoencoder(args):
    kf = KFold(n_splits=args.splits_num, shuffle=args.shuffle, random_state=42)

    score_lst = list()

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

        model = Autoencoder(number_of_users, number_of_movies, layers=args.hidden_layers, masking=args.masking)
        model.fit(data_zeros, data_mask, valid_users=valid_users, valid_movies=valid_movies, valid_ratings=valid_ratings, 
                    n_epochs=args.num_epochs, verbose=False)

        preds = model.predict(data_zeros, valid_users, valid_movies)

        score = root_mean_square_error(valid_ratings, preds)
        score_lst.append(score)

        print("Fold:", fold+1, "score:", score)

    print('Mean CV RMSE:', np.mean(score_lst))

def bias_sgd(args):
    # shuffle to ensure train examples exist for all users with high probability
    kf = KFold(n_splits=args.splits_num, shuffle=args.shuffle, random_state=42)

    score_lst = list()

    for fold, (train_index, valid_index) in enumerate(kf.split(users)):
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

        preds = model.predict(valid_users, valid_movies)

        score = root_mean_square_error(valid_ratings, preds)
        score_lst.append(score)

        print("Fold:", fold+1, "score:", score)

    print('Mean CV RMSE:', np.mean(score_lst))

def embeddings(args): 
    kf = KFold(n_splits=args.splits_num, shuffle=args.shuffle, random_state=42)

    score_lst = list()

    for fold, (train_index, valid_index) in enumerate(kf.split(users)):
        train_users = users[train_index]
        train_movies = movies[train_index]
        train_ratings = ratings[train_index]

        valid_users = users[valid_index]
        valid_movies = movies[valid_index]
        valid_ratings = ratings[valid_index]

        model = Embeddings(number_of_users, number_of_movies,
                       embeddings_size=args.embeddings_size,
                       dropout_embeddings=args.embeddings_dropout_embeddings,
                       dropout=args.embeddings_dropout)

        model.fit(train_users, train_movies, train_ratings,
              valid_users=valid_users, valid_movies=valid_movies,
              valid_ratings=valid_ratings, epochs=args.embeddings_num_epochs,
              verbose=args.verbose, decay=args.embeddings_decay,
              decay_steps=args.embeddings_decay_steps,
              learning_rate=args.embeddings_learning_rate,
              batch_size=args.embeddings_batch_size)

        preds = model.predict(valid_users, valid_movies)

        score = root_mean_square_error(valid_ratings, preds)
        score_lst.append(score)

        print("Fold:", fold+1, "score:", score)

    print('Mean CV RMSE:', np.mean(score_lst))

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    funs = dict(autoencoder=autoencoder, 
                bsgd=bias_sgd,
                isvd=iterative_svd,
                embeddins=embeddings)

    funs[args.model](args)
