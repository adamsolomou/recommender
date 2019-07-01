import numpy as np
import math
from sklearn.utils import shuffle
from utils import root_mean_square_error


def update_learning_rate(learning_rate, current_step, update_every, decay):
    return learning_rate * math.pow(decay, current_step / update_every)


# noinspection PyPep8Naming
class BiasSGD:
    def __init__(self,
                 number_of_users,
                 number_of_movies,
                 hidden_size=12,
                 regularization_matrix=0.08,
                 regularization_vector=0.04):

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

    def fit_data(self,
                 users_train,
                 movies_train,
                 ratings_train,
                 users_validation=None,
                 movies_validation=None,
                 ratings_validation=None,
                 num_epochs=50,
                 decay=1.5,
                 lr=0.05,
                 decay_every=5,
                 verbose=True):
        """
        Parameters
        ----------
        users_train             array of user ids for train
        movies_train            array of movie ids for train
        ratings_train           array of ratings for train
        users_validation        array of user ids for validation
        movies_validation       array of movie ids for validation
        ratings_validation      array of ratings for validation
        num_epochs              number of epochs
        decay                   divide the learning rate by this number every requested epochs
        lr                      initial learning rate
        decay_every             number of epoch every which to decay the learning rate
        verbose
        """

        validation = False
        if users_validation is not None and movies_validation is not None and ratings_validation is not None:
            validation = True


        average_rating = np.mean(ratings_train)

        for epoch in range(num_epochs):
            users_train_sh, movies_train_sh, ratings_train_sh = shuffle(users_train, movies_train, ratings_train)

            for user, movie, rating in zip(users_train_sh, movies_train_sh, ratings_train_sh):
                U_d = self.U[user, :]
                V_n = self.V[movie, :]

                biasU_d = self.biasU[user]
                biasV_n = self.biasV[movie]

                guess = U_d.dot(V_n) + biasU_d + biasV_n

                delta = rating - guess

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

            if validation and verbose:
                predictions = self.predict(users_validation, movies_validation)
                print('Validation error at epoch', epoch, 'is',
                      root_mean_square_error(ratings_validation, predictions))

            if (epoch + 1) % decay_every == 0:
                lr /= decay

    def predict(self, users_test, movies_test):

        predictions = list()

        for user, movie in zip(users_test, movies_test):
            predictions.append(self.U[user, :].dot(self.V[movie, :]) + self.biasU[user] + self.biasV[movie])

        return np.array(predictions)
