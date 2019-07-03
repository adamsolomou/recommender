import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from reader import fetch_data
from iterative_svd import IterativeSVD
from utils import root_mean_square_error

dataloader = fetch_data(train_size=0.9)
number_of_users, number_of_movies = 10000, 1000

IDs, users, movies, ratings, _ = dataloader['train']

def create_parser():
    parser = argparse.ArgumentParser(description="Run cross validation for model")
    parser.add_argument("--verbose", "-v", action="store_true")

    parser.add_argument("--splits-num", type=int, default=10)
    parser.add_argument("--shuffle", action="store_true", default=True)

    parser.add_argument("--shrinkage", type=int, default=38)
    parser.add_argument("--iterations", type=int, default=15)

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
        print("fold:", fold, "score:", score)