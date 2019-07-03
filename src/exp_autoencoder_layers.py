import numpy as np
import pandas as pd
import os
import tensorflow as tf
from utils import root_mean_square_error
from iterative_svd import IterativeSVD
from svdplus import SVDplus
from autoencoder import Autoencoder
from embeddings import Embeddings


device_name = tf.test.gpu_device_name()

print('Found GPU at: {}'.format(device_name))

DATA_DIR = '/cluster/home/saioanni/CIL/recommender/data/'
RANDOM_SEED = 42

number_of_users, number_of_movies = (10000, 1000)
testing=True
train_size = 0.88


data_train = pd.read_csv(os.path.join(DATA_DIR, 'data_train.csv'))

if not testing:
    submission = pd.read_csv(os.path.join(DATA_DIR, 'sampleSubmission.csv'))

# get average rating
average_rating = np.mean(data_train['Prediction'].values)

if testing:
    # Split the dataset into train and test
    np.random.seed(RANDOM_SEED)

    msk = np.random.rand(len(data_train)) < train_size
    train = data_train[msk]
    test = data_train[~msk]
else:
    train = data_train

users_train = list()
movies_train = list()
for user, movie in train.Id.str.extract('r(\d+)_c(\d+)').values:
    users_train.append(int(user) - 1)
    movies_train.append(int(movie) - 1)

ratings_train = train['Prediction'].values

if testing:
    users_test = list()
    movies_test = list()
    for user, movie in test.Id.str.extract('r(\d+)_c(\d+)').values:
        users_test.append(int(user) - 1)
        movies_test.append(int(movie) - 1)

    users_test = np.array(users_test)
    movies_test = np.array(movies_test)
    ratings_test = test['Prediction'].values
else:
    users_test = list()
    movies_test = list()
    for user, movie in submission.Id.str.extract('r(\d+)_c(\d+)').values:
        users_test.append(int(user) - 1)
        movies_test.append(int(movie) - 1)

users_train = np.array(users_train)
movies_train = np.array(movies_train)


# calculate average per movie
movie_ratings = {}

for i, (user, movie) in enumerate(zip(users_train, movies_train)):
    if movie in movie_ratings:
        movie_ratings[movie].append(ratings_train[i])
    else:
        movie_ratings[movie] = [ratings_train[i]]

movie_averages = np.zeros(number_of_movies)
for movie, ratings in movie_ratings.items():
    movie_averages[movie] = np.mean(ratings)


data_zeros = np.full((number_of_users, number_of_movies), 0)
data_average = np.tile(movie_averages, number_of_users).reshape(-1, number_of_movies)

data_mask = np.full((number_of_users, number_of_movies), 0)

for i, (user, movie) in enumerate(zip(users_train, movies_train)):
    data_zeros[user][movie] = ratings_train[i]
    data_average[user][movie] = ratings_train[i]
    data_mask[user][movie] = 1



for layer in [32, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]:
    model3 = Autoencoder(number_of_users, number_of_movies, layers=[layer], masking=0.5)
    model3.train(data_zeros, data_mask, users_validation=users_test, movies_validation=movies_test,
                 ratings_validations=ratings_test, n_epochs=200, verbose=False)

    preds = model3.predict(data_zeros, users_test, movies_test)
    score = root_mean_square_error(ratings_test, preds)
    print('layer: {} score: {}'.format(layer, score))
