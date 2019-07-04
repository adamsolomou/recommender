import sys
# sys.path.insert(0, '..')
sys.path.insert(0, './src')

import time
import argparse
import itertools

from pprint import pprint

import numpy as np
import pandas as pd

# from svdplus import SVDplus
from bias_sgd import BiasSGD
from reader import fetch_data
from utils import root_mean_square_error

N_USERS = 10000
N_MOVIES = 1000
N_ITERS = 1000000

#data = fetch_data(train_size=0.88, train_file="../data/data_train.csv",
#                  test_file="../data/sampleSubmission.csv")
data = fetch_data(train_size=0.88)

# Training 
_, train_users, train_movies, train_ratings, A_train = data['train']

# Validation 
_, valid_users, valid_movies, valid_ratings, A_valid = data['valid']


def predict_with_config(args, hidden_size=12, lr=0.04, reg_matrix=0.08,
                        reg_vector=0.04):
    predictor = BiasSGD(N_USERS, N_MOVIES, hidden_size=hidden_size,
                        regularization_matrix=reg_matrix,
                        regularization_vector=reg_vector)
    predictor.fit(train_users, train_movies, train_ratings,
                  valid_users=valid_users,
                  valid_movies=valid_movies,
                  valid_ratings=valid_ratings,
                  num_epochs=args.num_iter,
                  lr=lr)

    preds = predictor.predict(valid_users, valid_movies)
    err = root_mean_square_error(valid_ratings, preds)
    if args.sleep > 0:
        time.sleep(args.sleep)
    return err


def run_hidden_size_experiment(args):
    if args.verbose > 0:
        param_str = "Starting value: {}\nEnding value: {}"
        print("+=+=+=+=+Hidden Size parameters+=+=+=+=+")
        print(param_str.format(args.start, args.end))
        print("Running hidden size experiment...")
    hid_size_exp = {}
    for hidden_size in range(args.start, args.end + 1):
        print("hidden size: {}".format(hidden_size))
        print("\n")
        err = predict_with_config(args, hidden_size=hidden_size)
        hid_size_exp[hidden_size] = err
        print("Error: ", err)
    return hid_size_exp


def run_regularization_matrix_experiment(args):
    if args.verbose > 0:
        param_str = ("Starting value: {}\nEnding value: {}\n"
                     " Number of values: {}")
        print("+=+=+=+=+Regularisation Matrix parameters+=+=+=+=+")
        print(param_str.format(args.start, args.end, args.point_num))
    reg_matrix_exp = {}
    for reg_matrix in np.linspace(args.start, args.end, num=args.point_num):
        print("regularization matrix: {}".format(reg_matrix))
        print("\n")
        err = predict_with_config(args, reg_matrix=reg_matrix)
        reg_matrix_exp[reg_matrix] = err
        print("Error: ", err)
    return reg_matrix_exp


def run_regularization_vector_experiment(args):
    if args.verbose > 0:
        param_str = ("Starting value: {}\nEnding value: {}\n"
                     "Number of values: {}")
        print("+=+=+=+=+Regularisation Vector parameters+=+=+=+=+")
        print(param_str.format(args.start, args.end, args.point_num))
    reg_vector_exp = {}
    for reg_vector in np.linspace(args.start, args.end, num=args.point_num):
        print("regularization vector: {}".format(reg_vector))
        print("\n")
        err = predict_with_config(args, reg_vector=reg_vector)
        reg_vector_exp[reg_vector] = err
        print("Error: ", err)
    return reg_vector_exp

def run_grid_experiment(args):
    if args.verbose > 0:
        param_str = ("Starting reg value: {}\nEnding reg value: {}\n"
                     "Number of values: {}\n\nStarting hidden value: {}\n"
                     "Ending hidden value: {}")
        print("+=+=+=+=+Regularisation Vector parameters+=+=+=+=+")
        print(param_str.format(args.reg_start, args.reg_end,
                               args.reg_point_num,
                               args.hidden_start, args.hidden_end))
    points = itertools.product(range(args.hidden_start,
                                     args.hidden_end + 1),
                               np.linspace(args.reg_start,
                                           args.reg_end,
                                           num=args.reg_point_num))
    grid_exp = {}
    for hidden_size, reg in points:
        print("regularization vector: {}\nHidden size: {}".format(reg,
                                                                  hidden_size))
        print("\n")
        err = predict_with_config(args, reg_vector=reg,
                                  hidden_size=hidden_size)
        grid_exp[(hidden_size, reg)] = err
        print("Error: ", err)
    return grid_exp


def create_parser():
    parser = argparse.ArgumentParser(description="Run SVD+ experiments")
    parser.add_argument("--num-iter", type=int)
    parser.add_argument("--train-size", type=float)
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument("--sleep", type=int, default=0)
    
    subparsers = parser.add_subparsers(help="help msg")
    hid_exp_parser = subparsers.add_parser("hidden-size")
    hid_exp_parser.set_defaults(func=run_hidden_size_experiment)
    hid_exp_parser.add_argument("--start", type=int)
    hid_exp_parser.add_argument("--end", type=int)

    reg_matrix_exp_parser = subparsers.add_parser("reg-matrix")
    reg_matrix_exp_parser.set_defaults(func=run_regularization_matrix_experiment)
    reg_matrix_exp_parser.add_argument("--start", type=float)
    reg_matrix_exp_parser.add_argument("--end", type=float)
    reg_matrix_exp_parser.add_argument("--point-num", type=int)
    
    reg_vector_exp_parser = subparsers.add_parser("reg-vector")
    reg_vector_exp_parser.set_defaults(func=run_regularization_vector_experiment)
    reg_vector_exp_parser.add_argument("--start", type=float)
    reg_vector_exp_parser.add_argument("--end", type=float)
    reg_vector_exp_parser.add_argument("--point-num", type=int)
    
    grid_exp_parser = subparsers.add_parser("grid")
    grid_exp_parser.set_defaults(func=run_grid_experiment)
    grid_exp_parser.add_argument("--reg-start", type=float)
    grid_exp_parser.add_argument("--reg-end", type=float)
    grid_exp_parser.add_argument("--reg-point-num", type=int)
    grid_exp_parser.add_argument("--hidden-start", type=int)
    grid_exp_parser.add_argument("--hidden-end", type=int)
    
    return parser


if __name__ == "__main__":
    parser = create_parser()
    arguments = parser.parse_args()
    
    if arguments.verbose > 0:
        print("+=+=+=+=+Generial options+=+=+=+=+") 
        print("Number of iterations: {}".format(arguments.num_iter))
        print("Training set size (ratio): {}".format(arguments.train_size))

    result = arguments.func(arguments)
    pprint(result)
