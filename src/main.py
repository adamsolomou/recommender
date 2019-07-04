import argparse

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


def create_parser():
    parser = argparse.ArgumentParser(description="Run models")
    parser.add_argument("--verbose", "-v", action="store_true", default=False)

    parser.add_argument("--train-size", type=float, default=0.88)

    parser.add_argument("--shrinkage", type=int, default=38)
    parser.add_argument("--isvd-num-epochs", type=int, default=15)

    parser.add_argument("--hidden-size", type=int, default=12)
    parser.add_argument("--regularization_matrix", type=float, default=0.08)
    parser.add_argument("--regularization_vector", type=float, default=0.04)
    parser.add_argument("--sgd-num-epochs", type=int, default=50)
    parser.add_argument("--decay", type=float, default=1.5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--decay-every", type=int, default=5)

    parser.add_argument("--autoenc-regularization", type=float, default=0)
    parser.add_argument("--autoenc-num-epochs", type=int, default=0)
    parser.add_argument("--autoenc-masking", type=float, default=0.3)

    parser.add_argument("--embeddings-size", type=int, default=96)
    parser.add_argument("--embeddings-dropout-embeddings", type=float,
                        default=0.2)
    parser.add_argument("--embeddings-dropout", type=float, default=0.1)
    parser.add_argument("--embeddings-decay", type=float, default=0.97)
    parser.add_argument("--embeddings-learning-rate", type=float, default=1.0)
    parser.add_argument("--embeddings-decay-steps", type=int, default=2000)
    parser.add_argument("--embeddings-num-epochs", type=int, default=5)
    parser.add_argument("--embeddings-batch-size", type=int, default=512)

    parser.add_argument("--output-file", "-o",
                        default="ensemble_predictions.csv")

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        print('GPU device not found')
        print('Be sure you want to continue...')
    else:
        print('Found GPU at: {}'.format(device_name))

    dataloader = fetch_data(train_size=args.train_size)
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
    model = IterativeSVD(shrinkage=args.shrinkage)
    model.fit(A_train_average, known_train, A_valid=A_valid, valid_mask=known_validation, verbose=args.verbose, iterations=args.isvd_num_epochs)

    preds_iSVD_val = model.predict(valid_users, valid_movies)
    preds_iSVD_test = model.predict(test_users, test_movies)

    # biased SGD model
    model = BiasSGD(number_of_users=10000, number_of_movies=1000,
                    hidden_size=args.hidden_size,
                    regularization_matrix=args.regularization_matrix,
                    regularization_vector=args.regularization_vector)

    model.fit(train_users, train_movies, train_ratings,
              valid_users=valid_users, valid_movies=valid_movies,
              valid_ratings=valid_ratings,
              num_epochs=args.sgd_num_epochs, decay=args.decay, lr=args.lr,
              decay_every=args.decay_every, verbose=args.verbose)

    preds_bSGD_val = model.predict(valid_users, valid_movies)
    preds_bSGD_test = model.predict(test_users, test_movies)

    # autoencoder model
    model = Autoencoder(number_of_users, number_of_movies,
                        regularization=args.autoenc_regularization,
                        masking=args.autoenc_masking)
    model.fit(A_train_zeros, known_train_int, n_epochs=args.autoenc_num_epochs,
              valid_users=valid_users, valid_movies=valid_movies,
              valid_ratings=valid_ratings)

    preds = model.predict(A_train_zeros, test_users, test_movies)

    preds_autoencoder_val = model.predict(A_train_zeros, valid_users, valid_movies)
    preds_autoencoder_test = model.predict(A_train_zeros, test_users, test_movies)

    # embeddings model
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

    preds_embeddings_val = model.predict(valid_users, valid_movies)
    preds_embeddings_test = model.predict(test_users, test_movies)

    # create regressor for combined predictions
    valid_predictions = np.concatenate([preds_iSVD_val, preds_bSGD_val,
                                        preds_autoencoder_val,
                                        preds_autoencoder_val])

    valid_predictions = valid_predictions.reshape(4, valid_users.shape[0]).T

    test_predictions = np.concatenate([preds_iSVD_test, preds_bSGD_test, preds_autoencoder_test, preds_autoencoder_test])
    test_predictions = test_predictions.reshape(4, test_users.shape[0]).T

    regressor = LinearRegression(fit_intercept=True)
    regressor.fit(valid_predictions, valid_ratings)

    regressor_predictions = regressor.predict(test_predictions)

    # persist results
    preds_pd = pd.DataFrame(index=test_IDs, data=regressor_predictions, columns=['Prediction'])
    preds_pd.to_csv(args.output_file)
