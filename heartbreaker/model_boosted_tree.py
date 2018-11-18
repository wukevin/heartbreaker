import os
import sys
import itertools
import multiprocessing
import logging

import numpy as np
import pandas as pd
import sklearn

import xgboost

import data_loader
import util

def xgb(x_train, y_train, x_test, y_test, depth=6, n_est=250):
    """
    Train a boosted tree and return performance on testing data

    Note that the default hyperparameters listed in the function signature have been tuned
    to maximize performance via k fold cross validation. The default values provided by the
    library are:
    depth = 3
    n_est = 100
    """
    # Learning rate is default 0.1
    model = xgboost.XGBClassifier(max_depth=depth, learning_rate=1e-2, n_estimators=n_est, random_state=8292)
    logging.debug("Training XGBoost classifier with parameters depth={} n_estimators={}".format(depth, n_est))
    model.fit(x_train, y_train)
    # print(model)
    y_pred = model.predict(x_test)
    f1 = sklearn.metrics.f1_score(y_test, y_pred)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    precision = sklearn.metrics.precision_score(y_test, y_pred)
    recall = sklearn.metrics.recall_score(y_test, y_pred)
    return accuracy, precision, recall, f1

def main(percentile=25):
    data = util.impute_by_col(data_loader.load_all_data(), np.mean)
    rates = data.pop('heart_disease_mortality')
    rates_high_low = util.continuous_to_categorical(rates, 100 - percentile)
    train_validation_partitions, test_set = util.split_train_valid_k_fold(data, rates_high_low)

    # Evaluate k fold in parallel and tune hyperparameters
    parameters = []
    metrics = []
    depth_candidates = [4, 6, 8]
    num_est_candidates = [150, 200, 250, 300, 350]
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for depth, num_estimators in itertools.product(depth_candidates, num_est_candidates):
        performance_metrics = pool.starmap(xgb, [part + (depth, num_estimators) for part in train_validation_partitions])
        overall = np.vstack(performance_metrics)
        logging.info("Average XGBoost metrics with depth={} n_estimators={}: {}".format(depth, num_estimators, np.mean(overall, axis=0)))
        parameters.append((depth, num_estimators))
        metrics.append(np.mean(overall, axis=0))
    pool.close()
    pool.join()

    metrics = np.vstack(metrics)
    for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1 Score']):
        best_index = np.argmax(metrics[:,i])
        logging.info("Model with best {metric}:\t{value}\t{hyperparams}".format(
            metric=metric,
            value=np.round(metrics[best_index, i], decimals=4),
            hyperparams=parameters[best_index],
        ))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
