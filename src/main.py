import numpy as np
import pandas as pd
import os
import tensorflow as tf


device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')

print('Found GPU at: {}'.format(device_name))


DATA_DIR = None
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

train_users = list()
train_movies = list()
for user, movie in train.Id.str.extract('r(\d+)_c(\d+)').values:
    train_users.append(int(user) - 1)
    train_movies.append(int(movie) - 1)


if testing:
    test_users = list()
    test_movies = list()
    for user, movie in test.Id.str.extract('r(\d+)_c(\d+)').values:
        test_users.append(int(user) - 1)
        test_movies.append(int(movie) - 1)

    test_users = np.array(test_users)
    test_movies = np.array(test_movies)
else:
    test_users = list()
    test_movies = list()
    for user, movie in submission.Id.str.extract('r(\d+)_c(\d+)').values:
        test_users.append(int(user) - 1)
        test_movies.append(int(movie) - 1)

train_users = np.array(train_users)
train_movies = np.array(train_movies)


# calculate average per movie
movie_ratings = {}

for i, (user, movie) in enumerate(zip(train_users, train_movies)):
    if movie in movie_ratings:
        movie_ratings[movie].append(train['Prediction'].values[i])
    else:
        movie_ratings[movie] = [train['Prediction'].values[i]]

movie_averages = np.zeros(number_of_movies)
for movie, ratings in movie_ratings.items():
    movie_averages[movie] = np.mean(ratings)


data = np.full((number_of_users, number_of_movies), 0)

data_mask = np.full((number_of_users, number_of_movies), 0)

for i, (user, movie) in enumerate(zip(train_users, train_movies)):
    data[user][movie] = train['Prediction'].values[i]
    data_mask[user][movie] = 1

+ make initilizations with averages and zeros ....