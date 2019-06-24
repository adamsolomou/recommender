import numpy as np


class IterativeSVD:
    data_iterative_svd = None

    def __init__(self,
                 number_of_users,
                 number_of_movies,
                 singular_values_to_keep=None):

        if singular_values_to_keep is None:
            self.singular_values_to_keep = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 6, 10]
        else:
            self.singular_values_to_keep = singular_values_to_keep

        self.maximum_rank = min(number_of_users, number_of_movies)
        self.number_of_users = number_of_users
        self.number_of_movies = number_of_movies

    # noinspection PyPep8Naming
    def fit_data(self,
                 data,
                 users_train,
                 movies_train,
                 ratings_train):

        self.data_iterative_svd = data.copy()

        for singular_values in self.singular_values_to_keep:

            U, s, Vt = np.linalg.svd(self.data_iterative_svd, full_matrices=True)

            S = np.zeros((self.number_of_users, self.number_of_movies))

            S[:self.maximum_rank, :self.maximum_rank] = np.diag(s)

            Sk = S.copy()
            Sk[singular_values:, singular_values:] = 0

            self.data_iterative_svd = U.dot(Sk).dot(Vt)
            
            # fill in true values
            for (user, movie, rating) in zip(users_train, movies_train, ratings_train):
                self.data_iterative_svd[user][movie] = rating

            # fill in true values
            for (user, movie, rating) in zip(users_train, movies_train, ratings_train):
                self.data_iterative_svd[user][movie] = rating

    def predict(self,
                users_test,
                movies_test):

        assert self.data_iterative_svd is not None, "You first have to fit the data"

        predictions = np.zeros(len(users_test))

        for i, (user, movie) in enumerate(zip(users_test, movies_test)):
            predictions[i] = self.data_iterative_svd[user][movie]

        return predictions

