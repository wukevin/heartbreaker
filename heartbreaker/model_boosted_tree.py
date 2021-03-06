import os
import sys
import itertools
import multiprocessing
import logging

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import class_weight

import xgboost

import data_loader
import util
import plotting
from classification_v2 import get_gscv, adjust_params

seed = 754927

def xgb(x_train, y_train, x_test, y_test, depth=8, n_est=350):
    """
    Train a boosted tree and return performance on testing data

    Note that the default hyperparameters listed in the function signature have been tuned
    to maximize performance via k fold cross validation. The default values provided by the
    library are:
    depth = 3
    n_est = 100
    """
    # Learning rate is default 0.1
    model = xgboost.XGBClassifier(max_depth=depth, learning_rate=1e-2, n_estimators=n_est, random_state=8292, class_weights='balanced')
    logging.debug("Training XGBoost classifier with parameters depth={} n_estimators={}".format(depth, n_est))
    model.fit(x_train, y_train)
    # print(model)
    y_pred = model.predict(x_test)
    f1 = sklearn.metrics.f1_score(y_test, y_pred)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    precision = sklearn.metrics.precision_score(y_test, y_pred)
    recall = sklearn.metrics.recall_score(y_test, y_pred)
    return accuracy, precision, recall, f1

def parameter_sweep(percentile=25, depth_candidates=[4, 6, 8], num_est_candidates=[150, 200, 250, 300, 350]):
    """Sweet the hyperparameters and see what is the best combination"""
    data = util.impute_by_col(data_loader.load_all_data(), np.mean)
    rates = data.pop('heart_disease_mortality')
    rates_high_low = util.continuous_to_categorical(rates, 100 - percentile)
    train_validation_partitions, test_set = util.split_train_valid_k_fold(data, rates_high_low)

    # Evaluate k fold in parallel and tune hyperparameters
    parameters = []
    metrics = []
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

def parameter_sweep_pipeline(percentile=25, depth_candidates=[3, 4, 6, 8], num_est_candidates=[150, 200, 250, 300, 350, 400, 450], seed=8558):
    data = util.impute_by_col(data_loader.load_all_data(), np.mean)
    rates = data.pop('heart_disease_mortality')
    rates_discrete = util.continuous_to_categorical(rates, percentile_cutoff=75)
    x = data
    y = rates_discrete
    x_train, x_test, y_train, y_test = train_test_split(x, rates_discrete, test_size=0.10, random_state=seed)

    weights = class_weight.compute_class_weight('balanced', [0, 1], y_train)
    weight_ratio = weights[1] / weights[0]
    logging.info("XGBoost weight ratio: {}".format(np.round(weight_ratio, 4)))

    models = []
    boosted_tree = xgboost.XGBClassifier(scale_pos_weight=weight_ratio, random_state=8292, reg_lambda=0)
    boosted_tree_params = {
        "max_depth": depth_candidates,
        "n_estimators": num_est_candidates,
        "reg_alpha": [0.001, 0.01, 0.1, 1, 10, 100],
        "learning_rate": [1e-2, 1e-1],
        "booster": ["gbtree", "gblinear", "dart"],
        "min_child_weight": [1, 2, 3, 4],
    }
    models.append((boosted_tree, boosted_tree_params))

    for (model, params) in models:
        grid_params = adjust_params(params)
        cv = get_gscv(model, grid_params, scale=False, preselect_features=False)
        cv.fit(x_train, y_train)

        logging.info(model.__class__.__name__)
        logging.info("Best F1 Score")
        logging.info(cv.best_score_)
        logging.info("Best Parameters")
        logging.info(cv.best_params_)

def eval_on_test_set(booster='gbtree', lr=0.1, max_depth=6, min_child_weight=1, n_estimators=150, reg_alpha=10):
    """The params in the above are from expanded grid search"""
    data = util.impute_by_col(data_loader.load_all_data(), np.mean)
    rates = data.pop('heart_disease_mortality')
    rates_discrete = util.continuous_to_categorical(rates, percentile_cutoff=75)
    x = data
    y = rates_discrete
    x_train, x_test, y_train, y_test = train_test_split(x, rates_discrete, test_size=0.10, random_state=seed)

    weights = class_weight.compute_class_weight('balanced', [0, 1], y_train)
    weight_ratio = weights[1] / weights[0]
    logging.info("XGBoost weight ratio: {}".format(np.round(weight_ratio, 4)))

    model = xgboost.XGBClassifier(
        booster=booster,
        lr=lr,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        n_estimators=n_estimators,
        reg_alpha=reg_alpha,
        scale_pos_weight=weight_ratio,
        random_state=8292,
        reg_lambda=0,
    )

    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    
    logging.info("Test Accuracy: {}".format(accuracy_score(y_test, preds)))
    logging.info("Test Precision: {}".format(precision_score(y_test, preds)))
    logging.info("Test Recall: {}".format(recall_score(y_test, preds)))
    logging.info("Test F1 score: {}".format(f1_score(y_test, preds)))
    logging.info("Confusion matrix:\n{}".format(confusion_matrix(y_test, preds)))

def eval_on_train_set(booster='gbtree', lr=0.1, max_depth=6, min_child_weight=1, n_estimators=150, reg_alpha=10):
    """The params in the above are from expanded grid search"""
    data = util.impute_by_col(data_loader.load_all_data(), np.mean)
    rates = data.pop('heart_disease_mortality')
    rates_discrete = util.continuous_to_categorical(rates, percentile_cutoff=75)
    x = data
    y = rates_discrete
    partitions, holdout = util.split_train_valid_k_fold(x, y)

    all_truths = []
    all_preds = []
    for part in partitions:
        x_train, y_train, x_test, y_test = part
        weights = class_weight.compute_class_weight('balanced', [0, 1], y_train)
        weight_ratio = weights[1] / weights[0]
        logging.info("XGBoost weight ratio: {}".format(np.round(weight_ratio, 4)))

        model = xgboost.XGBClassifier(
            booster=booster,
            lr=lr,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            n_estimators=n_estimators,
            reg_alpha=reg_alpha,
            scale_pos_weight=weight_ratio,
            random_state=8292,
            reg_lambda=0,
        )
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        all_truths.extend(y_test)
        all_preds.extend(preds)
    
    logging.info("Train Accuracy: {}".format(accuracy_score(all_truths, all_preds)))
    logging.info("Train Precision: {}".format(precision_score(all_truths, all_preds)))
    logging.info("Train Recall: {}".format(recall_score(all_truths, all_preds)))
    logging.info("Train F1 score: {}".format(f1_score(all_truths, all_preds)))

def feature_importance(percentile=25):
    """Evaluate feature importance by fitting a model to ALL the data"""
    data = util.impute_by_col(data_loader.load_all_data(), np.mean)
    rates = data.pop('heart_disease_mortality')
    rates_high_low = util.continuous_to_categorical(rates, 100 - percentile)

    x_train, x_test, y_train, y_test = train_test_split(data, rates_high_low, test_size=0.10, random_state=seed)

    weights = class_weight.compute_class_weight('balanced', [0, 1], y_train)
    weight_ratio = weights[1] / weights[0]
    model = xgboost.XGBClassifier(booster='gbtree', lr=0.1, max_depth=6, min_child_weight=1, n_estimators=150, reg_alpha=10, reg_lambda=0, random_state=8292)  # 6 and 250 for depth and n_estimators were found via parameter sweep
    model.fit(data, rates_high_low)

    plotting.plot_shap_tree_summary(model, x_train, x_test, output_fname=os.path.join(plotting.PLOTS_DIR, "shap_xgboost_importance.png"))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # parameter_sweep()
    # parameter_sweep_pipeline()
    eval_on_train_set()
    eval_on_test_set()
    # feature_importance()
