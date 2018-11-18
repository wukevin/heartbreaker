"""
Code for an SVM regression

When executed, prints out the R^2 value and the MSE values to stderr
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
import sklearn
from sklearn import svm
from sklearn import preprocessing 

import data_loader
import util

def svr(x_train, y_train, x_test, y_test):
    """Train a least squares model and return performance on testing data"""
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = pd.DataFrame(scaler.transform(x_train), columns=x_train.columns)
    model = svm.SVR(gamma='scale', kernel='sigmoid')
    model.fit(x_train, y_train)
    x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)
    predictions = model.predict(x_test)
    logging.info("SVR R^2 value: {}".format(model.score(x_test, y_test)))
    print(model.score(x_test, y_test))
    mse = sklearn.metrics.mean_squared_error(y_test, predictions)
    logging.info("SVR MSE: {}".format(mse))
    return mse

def main():
    """Run the script"""
    data = util.impute_by_col(data_loader.load_all_data(), np.mean)
    rates = data.pop('heart_disease_mortality')
    partitions = util.split_test_train_k_fold(data, rates)
    logging.info("Running MSE on first partition")
    first_set = partitions[0]
    mse = svr(*first_set)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
