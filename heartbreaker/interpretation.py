"""
Code for doing ML model interpretation
"""

import os
import sys
import logging
import multiprocessing

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sklearn.metrics

import data_loader
import util

def fit_and_predict(model, x_train, y_train, x_test):
    """Take a model that has .fit and .predict methods, train it, and return its predictions"""
    model.fit(x_train, y_train)
    return model.predict(x_test)

def feature_backwards_search(x_train, y_train, x_test, y_test, fname, model, **kwargs):
    """
    Do a depth-of-one feature backwards search. Does not assume that the input is standardized.
    kwargs are passed to the model initialization. Returns versions of x_train and x_test with
    the highest scores. If we can drop a column to improve performance, we drop it and return the
    subset; if not, we simply return x_train and x_test as given.
    """
    assert isinstance(fname, str)
    # Standardize
    sc = StandardScaler()
    x_train_std = sc.fit_transform(x_train)
    x_test_std = sc.transform(x_test)
    assert x_train_std.shape[1] == x_test_std.shape[1]
    
    # Create subsetted input data sets
    logging.info("Evaluating {} features".format(x_train_std.shape[1]))
    x_train_std_dropped = [np.delete(x_train_std, i, 1) for i in range(x_train_std.shape[1])]
    x_test_std_dropped = [np.delete(x_test_std, i, 1) for i in range(x_test_std.shape[1])]
    for training_set in x_train_std_dropped:  # sanity check
        assert training_set.shape == (training_set.shape[0], x_train_std.shape[1] - 1)
    
    # Create a bunch of models, one for each dropped dataset, using kwargs
    models = [model(**kwargs) for _i in range(x_train_std.shape[1])]
    
    # Fit each in parallel
    pool = multiprocessing.Pool(int(multiprocessing.cpu_count()))
    predictions = pool.starmap(fit_and_predict, [(model, train_data, y_train, test_data) for model, train_data, test_data in zip(models, x_train_std_dropped, x_test_std_dropped)])
    pool.close()
    pool.join()

    # Evaluate them
    baseline_predictions = fit_and_predict(model(**kwargs), x_train_std, y_train, x_test_std)
    baseline_score = sklearn.metrics.f1_score(y_test, baseline_predictions)
    logging.info("Baseline score with full data: {}".format(baseline_score))
    dropped_scores = [sklearn.metrics.f1_score(y_test, preds) for preds in predictions]
    logging.info("Min and max scores after dropping features: {} {}".format(min(dropped_scores), max(dropped_scores)))

    if fname:
        raise NotImplementedError()  # Supposed to write out the relative importance here

    # Drop column whose loss results in the largest improvement in performance
    if max(dropped_scores) > baseline_score:
        best_index_to_drop = np.argmax(dropped_scores)
        best_column_to_drop = x_train.columns[best_index_to_drop]
        logging.info("Dropping {}".format(best_column_to_drop))
        x_train_sub = x_train.drop(columns=best_column_to_drop)
        x_test_sub = x_test.drop(columns=best_column_to_drop)
        return x_train_sub, x_test_sub
    return x_train, x_test

def main(percentile=25):
    """Execute script"""
    data = util.impute_by_col(data_loader.load_all_data())
    rates = data.pop('heart_disease_mortality')
    rates_high_low = util.continuous_to_categorical(rates, 100 - percentile)
    train_validation_partitions, test_set = util.split_train_valid_k_fold(data, rates_high_low)

    # Use the first split
    x_train, y_train, x_test, y_test = train_validation_partitions[0]
    feature_backwards_search(x_train, y_train, x_test, y_test, "", LogisticRegression,
        solver='liblinear', class_weight='balanced', C=1, penalty='l1', random_state=98572, max_iter=1000)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
