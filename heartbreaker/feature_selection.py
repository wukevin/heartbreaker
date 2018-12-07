"""
Code for doing ML model interpretation
"""

import os
import sys
import logging
import multiprocessing

import numpy as np
import pandas as pd
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
        df = pd.DataFrame(
            dropped_scores,
            columns=['F1'],
            index=x_train.columns,
        )
        df.at['full_baseline'] = baseline_score
        df.to_csv(fname)

    # Drop column whose loss results in the largest improvement in performance
    if max(dropped_scores) > baseline_score:
        best_index_to_drop = np.argmax(dropped_scores)
        best_column_to_drop = x_train.columns[best_index_to_drop]
        logging.info("Dropping {}".format(best_column_to_drop))
        x_train_sub = x_train.drop(columns=best_column_to_drop)
        x_test_sub = x_test.drop(columns=best_column_to_drop)
        return x_train_sub, x_test_sub
    return x_train, x_test

def feature_forward_search(partitions, num_features, model, **kwargs):
    """Performs forward search on features"""
    def extract_features_from_partition(partition, feature_names):
        """Given a single partition, recreate it with only the desired feature names"""
        x_train, y_train, x_test, y_test = partition
        x_train_sub = x_train.loc[:, feature_names]
        x_test_sub = x_test.loc[:, feature_names]
        assert x_train_sub.shape[1] == x_test_sub.shape[1] == len(feature_names)
        return x_train_sub, y_train, x_test_sub, y_test

    all_feature_names = list(partitions[0][0].columns)
    selected_feature_names = []
    selected_feature_scores = []

    pool = multiprocessing.Pool(int(multiprocessing.cpu_count() / 2))
    while len(selected_feature_names) < min(num_features, len(all_feature_names)):
        # Define which features we're considering
        candidate_features = [feature for feature in all_feature_names if feature not in selected_feature_names]
        assert len(candidate_features) + len(selected_feature_names) == len(all_feature_names)
        # Create a model for each
        models = [model(**kwargs) for _i in range(len(candidate_features))]
        # Trim the cross validation partitions to include only selected features
        partitions_subfeatures = []
        for ft in candidate_features:
            partitions_subfeatures.append([extract_features_from_partition(part, selected_feature_names + [ft]) for part in partitions])
        # Perform cross validation and evaluate them
        results = pool.starmap(util.cross_validate, [(part, model) for part, model in zip(partitions_subfeatures, models)])
        results_metrics = [sklearn.metrics.f1_score(*pair) for pair in results]
        best_feature = candidate_features[np.argmax(results_metrics)]
        logging.info("{}: adding feature {}".format(np.round(max(results_metrics), 4), best_feature))
        selected_feature_names.append(best_feature)
        selected_feature_scores.append(max(results_metrics))
    pool.close()
    pool.join()

    return selected_feature_names, selected_feature_scores

def run_forward_search(num_features=50, percentile=25):
    """Execute script"""
    data = util.impute_by_col(data_loader.load_all_data())
    rates = data.pop('heart_disease_mortality')
    rates_high_low = util.continuous_to_categorical(rates, 100 - percentile)
    train_validation_partitions, test_set = util.split_train_valid_k_fold(data, rates_high_low)

    if num_features <= 0:
        num_features = data.shape[1]  # Set the number of features to be *everything*
        logging.info("Got a value <= 0 for num_features; evaluating all {} features".format(num_features))
    features, scores = feature_forward_search(train_validation_partitions, num_features, LogisticRegression,
        solver='liblinear', class_weight='balanced', C=1, penalty='l1', random_state=98572, max_iter=1000)
    df = pd.DataFrame(scores, index=features, columns=['F1'])
    df.to_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "results/forward_search_{}_logreg.csv".format(num_features)))

def run_backwards_search(percentile=25):
    """Execute script"""
    data = util.impute_by_col(data_loader.load_all_data())
    rates = data.pop('heart_disease_mortality')
    rates_high_low = util.continuous_to_categorical(rates, 100 - percentile)
    train_validation_partitions, test_set = util.split_train_valid_k_fold(data, rates_high_low)

    # Use the first split
    x_train, y_train, x_test, y_test = train_validation_partitions[0]

    # Perform backwards set until we have a final set
    x_train_old, x_test_old = None, None
    i = 0
    while x_train_old is not x_train:
        x_train_old = x_train
        x_test_old = x_test
        x_train, x_test = feature_backwards_search(x_train, y_train, x_test, y_test, os.path.join(os.path.dirname(os.path.dirname(__file__)), "results/backward_search_logreg_{}.csv".format(i)), LogisticRegression,
            solver='liblinear', class_weight='balanced', C=1, penalty='l1', random_state=98572, max_iter=1000)
        i += 1

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) == 2:
        run_forward_search(int(sys.argv[1]))
    else:
        run_forward_search()
