import numpy as np


class IterativeSVD(object):
    X = None

    def __init__(self, shrinkage=38):
        """
        Parameters:
        -----------
        k_per_iter : integer list
            Each entry corresponds to the number of singular values to keep
            when truncating the SVD.
        """
        self.shrinkage = shrinkage

    # noinspection PyPep8Naming
    def fit(self, A_train, train_mask, A_valid=None, valid_mask=None, iterations=15, verbose=True):
        """
        Fit the training data.
        Parameters:
        -----------
        A_train : array-like, shape=(n_users, n_movies)
            The imputed user-item training matrix.
        train_mask: array-like, shape=(n_users, n_movies)
            A boolean mask with True endings for known training ratings and False for unkown.
        A_valid : array-like, shape=(n_users, n_movies)
            The imputed user-item validation matrix.
        valid_mask: array-like, shape=(n_users, n_movies)
            A boolean mask with True endings for known validation ratings and False for unkown.
        Returns:
        --------
        If A_valid and valid_mask are provided a list with the error on the validation set for
        each iteration is returned. Otherwise nothing is returned.
        """

        # Validation mode
        validation = False
        if A_valid is not None and valid_mask is not None:
            validation = True
            valid_error_per_iter = list()

        # Make a copy
        self.X = A_train.copy()

        for iter_ in range(iterations):
            # SVD
            U, s, Vt = np.linalg.svd(self.X, full_matrices=False)

            # Truncate SVD
            s_ = (s - self.shrinkage).clip(min=0)

            # Reconstruct
            self.X = np.dot(np.dot(U, np.diag(s_)), Vt)

            # Restore observed ratings
            self.X[train_mask] = A_train[train_mask]

            # Validation mode
            if validation and verbose:
                error = A_valid[valid_mask] - self.X[valid_mask]
                valid_error = np.sqrt(np.mean(np.square(error)))
                # noinspection PyUnboundLocalVariable
                valid_error_per_iter.append(valid_error)

        if validation:
            return valid_error_per_iter

    def predict(self, test_users, test_movies):
        """
        Predict ratings for test_users and test_movies
        Parameters:
        -----------
        test_users : array-like, shape=(test_size,)
            The user ID for each required prediction.
        test_movies : array-like, shape=(test_size,)
            The movie ID for each required prediction.
        Returns:
        --------
        predictions : array-like, shape=(test_size,)
            Predicted ratings for each (user,movie) pair
        """

        assert self.X is not None, "You first have to fit the data."

        predictions = np.zeros(len(test_users))

        for i, (user, movie) in enumerate(zip(test_users, test_movies)):
            predictions[i] = self.X[user, movie]

        return predictions
