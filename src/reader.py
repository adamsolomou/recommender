import numpy as np
import pandas as pd


def fetch_data(train_size=1.0, n_users=10000, n_movies=1000, train_file='data_train.csv',
               test_file='sampleSubmission.csv', seed=42):
    """
    Load the dataset file
    Parameters:
    -----------
    train_size : float
        The portion of the dataset to include in the training split.
    n_users : int
        The number of users
    n_movies : int
        The number of items
    train_file : string
        Name of the training set file
    test_file : string
        Name of the testing set file
    seed : int
        Seed
    Returns:
    --------
    dataloader : dict
        'train': User IDs, Movie IDs, Ratings, Rating matrix (missing entries set to NaN)
        'valid': If train_size = 1.0 then None, otherwise same format as training dataloader
        'test' : User IDs, Movie IDs
    """
    # Fix the random seed
    np.random.seed(seed)

    # Load the data
    data = pd.read_csv('data/%s' % train_file)
    test_data = pd.read_csv('data/%s' % test_file)

    validation = False

    if train_size < 1.0:
        validation = True

        # Train-validation split
        msk = np.random.rand(len(data)) < train_size

        train_data = data[msk]
        valid_data = data[~msk]
    else:
        train_data = data

    ################### Training data ###################
    train_users = list()
    train_movies = list()

    train_IDs = train_data.Id.values
    train_ratings = train_data['Prediction'].values

    A_train = np.zeros(shape=(n_users, n_movies))
    A_train[:] = np.nan

    for i, (u_idx, m_idx) in enumerate(train_data.Id.str.extract('r(\d+)_c(\d+)').values):
        # Cast
        u_idx = int(u_idx) - 1
        m_idx = int(m_idx) - 1

        train_users.append(u_idx)
        train_movies.append(m_idx)

        A_train[u_idx, m_idx] = train_ratings[i]

    train_users = np.array(train_users)
    train_movies = np.array(train_movies)

    ################### Validation dataset ###################
    if validation:
        valid_users = list()
        valid_movies = list()

        valid_IDs = valid_data.Id.values
        valid_ratings = valid_data['Prediction'].values

        A_valid = np.zeros(shape=(n_users, n_movies))
        A_valid[:] = np.nan

        for i, (u_idx, m_idx) in enumerate(valid_data.Id.str.extract('r(\d+)_c(\d+)').values):
            # Cast
            u_idx = int(u_idx) - 1
            m_idx = int(m_idx) - 1

            valid_users.append(u_idx)
            valid_movies.append(m_idx)

            A_valid[u_idx, m_idx] = valid_ratings[i]

        valid_users = np.array(valid_users)
        valid_movies = np.array(valid_movies)

    ################### Testing dataset ###################
    test_users = list()
    test_movies = list()

    test_IDs = test_data.Id.values

    for u_idx, m_idx in test_data.Id.str.extract('r(\d+)_c(\d+)').values:
        # Cast
        u_idx = int(u_idx) - 1
        m_idx = int(m_idx) - 1

        test_users.append(u_idx)
        test_movies.append(m_idx)

    test_users = np.array(test_users)
    test_movies = np.array(test_movies)

    if validation:
        dataloader = {'train': (train_IDs, train_users, train_movies, train_ratings, A_train),
                      'valid': (valid_IDs, valid_users, valid_movies, valid_ratings, A_valid),
                      'test': (test_IDs, test_users, test_movies)}
    else:
        dataloader = {'train': (train_IDs, train_users, train_movies, train_ratings, A_train),
                      'valid': None,
                      'test': (test_IDs, test_users, test_movies)}

    return dataloader
