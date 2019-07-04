import sys
sys.path.insert(0, '..')

import numpy as np
import math
from random import randint

from utils import root_mean_square_error


def update_learning_rate(learning_rate, current_step, update_every, decay):
    return learning_rate * math.pow(decay, current_step / update_every)


# noinspection PyPep8Naming
class SVDplus:
    def __init__(self,
                 number_of_users,
                 number_of_movies,
                 hidden_size=12,
                 regularization_matrix=0.08,
                 regularization_vector=0.04,
                 verbose=0):

        self.number_of_users = number_of_users
        self.number_of_movies = number_of_movies
        self.hidden_size = hidden_size
        self.regularization_matrix = regularization_matrix
        self.regularization_vector = regularization_vector

        # initialize matrices U and V
        def initialize(d0, d1, minimum=-0.02, maximum=0.02):
            return np.random.rand(d0, d1) * (maximum - minimum) + minimum

        self.U = initialize(self.number_of_users, self.hidden_size)
        self.V = initialize(self.number_of_movies, self.hidden_size)

        self.biasU = np.zeros(self.number_of_users)
        self.biasV = np.zeros(self.number_of_movies)

        self.verbose = verbose

    def fit_data(self,
                 users_train,
                 movies_train,
                 ratings_train,
                 users_validation=None,
                 movies_validation=None,
                 ratings_validation=None,
                 num_iters=150000000,
                 update_every=1000000,
                 learning_rate=0.04,
                 decay=0.95,
                 verbose=True):

        validation = False
        if users_validation is not None and movies_validation is not None and ratings_validation is not None:
            validation = True

        # for better parameter tuning please refer to past shown report (uploaded by Adamos)
        # for the moment pure stochastic
        # also make this in epochs by taking user-movie-rating sequencially and shuffling each epoch
        step = 0

        average_rating = np.mean(ratings_train)

        for iteration in range(num_iters):
            step += 1
            lr = update_learning_rate(learning_rate, step, update_every, decay)

            index = randint(0, len(users_train) - 1)
            user = users_train[index]
            movie = movies_train[index]

            U_d = self.U[user, :]
            V_n = self.V[movie, :]

            biasU_d = self.biasU[user]
            biasV_n = self.biasV[movie]

            guess = U_d.dot(V_n) + biasU_d + biasV_n

            delta = ratings_train[index] - guess

            try:
                new_U_d = U_d + lr * (delta * V_n - self.regularization_matrix * U_d)
                new_V_n = V_n + lr * (delta * U_d - self.regularization_matrix * V_n)

                new_biasU_d = biasU_d + lr * (delta - self.regularization_vector * (biasU_d + biasV_n - average_rating))
                new_biasV_n = biasV_n + lr * (delta - self.regularization_vector * (biasV_n + biasU_d - average_rating))

            except FloatingPointError:
                continue
            else:
                self.U[user, :] = new_U_d
                self.V[movie, :] = new_V_n

                self.biasU[user] = new_biasU_d
                self.biasV[movie] = new_biasV_n

            if self.verbose > 0:
                if iteration % 1000000 == 0 and validation and verbose:
                    predictions = self.predict(users_validation,
                                               movies_validation)
                    print('Validation error at iteration', iteration, 'is',
                          root_mean_square_error(ratings_validation,
                                                 predictions))

    def predict(self, users_test, movies_test):

        predictions = list()

        for user, movie in zip(users_test, movies_test):
            predictions.append(self.U[user, :].dot(self.V[movie, :]) + self.biasU[user] + self.biasV[movie])

        return np.array(predictions)
