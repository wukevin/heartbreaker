"""
Code for a linear regression as a very basic baseline

When executed, prints out the R^2 value and the MSE values to stderr
"""

import os
import sys
import multiprocessing
import logging

import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model

import data_loader
import util

def least_squares(x_train, y_train, x_test, y_test):
    """Train a least squares model and return performance on testing data"""
    model = sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=-1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    logging.info("Linear regression R^2 value: {}".format(model.score(x_test, y_test)))
    mse = sklearn.metrics.mean_squared_error(y_test, predictions)
    logging.info("Linear regression MSE: {}".format(mse))
    return mse

def main():
    """Run the script"""
    data = util.impute_by_col(data_loader.load_all_data(), np.mean)
    rates = data.pop('heart_disease_mortality')
    train_validation_partitions, test_set = util.split_train_valid_k_fold(data, rates)

    # Evaluate on the training/validation partitions in parallel
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    mse_values = pool.starmap(least_squares, train_validation_partitions)
    pool.close()
    pool.join()

    # Average the results
    logging.info("Average of {} cross validation runs: {}".format(len(mse_values), np.mean(mse_values)))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
