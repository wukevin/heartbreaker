"""
Logistic regression
"""

import os
import sys
import itertools
import multiprocessing
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import xgboost

import data_loader
import util
import plotting

plt.rcParams['figure.figsize'] = [18, 9]

def plot_overall_feature_contributions(df, weights, fname):
    """
    Plots overall feature contributions for a table with features on the left, observations on top
    Weights should be a dictionary or pandas series that supports weights[label]
    """
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == len(weights)
    multiplied = []
    for _i, row in df.iterrows():
        multiplied.append([ft * w][0] for ft, w in zip(row, weights))
    multiplied = pd.DataFrame(multiplied, index=df.index, columns=df.columns).transpose()
    multiplied = multiplied[weights != 0]
    weights = weights[weights != 0]
    medians = multiplied.median(1)
    assert len(medians) == multiplied.shape[0]
    medians.sort_values(ascending=False, inplace=True)
    multiplied = multiplied.loc[medians.index]
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[4, 1]}, sharex=True)
    ax1.boxplot(multiplied)
    ax1.set_ylabel("Contribution to prediction (weight * feature value)")
    ax1.set_title("Aggregated feature contributions")
    
    weights_reordered = np.array([weights[label] for label in medians.index])
    assert len(weights_reordered) == len(medians.index)
    ax2.bar(np.arange(len(weights_reordered)) + 1, weights_reordered)  # +1 because boxplot labels are 1-indexed
    ax2.set_xticklabels(multiplied.index, rotation=90)
    ax2.set_xlabel("Features (n={})".format(len(weights_reordered)))
    ax2.set_ylabel("Log. reg. weight")

    plt.savefig(fname, bbox_inches='tight', dpi=600)
    # print(fname)

def log_reg(x_train, y_train, x_test, y_test, c=0.1, ft_contrib_plot_fname=""):
    """
    Train logistic regression with L1 regularization. C value in function signature is optimal
    value based on parameter sweep.
    """
    # Standardize
    sc = StandardScaler()
    x_train_std = sc.fit_transform(x_train)
    x_test_std = sc.transform(x_test)
    model = LogisticRegression(penalty='l1', max_iter=1000, solver='liblinear', C=c, random_state=98572)
    logging.debug("Training logistic regression with parameters c={}".format(c))
    model.fit(x_train_std, y_train)
    y_pred = model.predict(x_test_std)
    f1 = sklearn.metrics.f1_score(y_test, y_pred)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    precision = sklearn.metrics.precision_score(y_test, y_pred)
    recall = sklearn.metrics.recall_score(y_test, y_pred)

    fn_indices = [index for index in range(len(y_pred)) if y_test[index] and not y_pred[index]]
    fn_feature_matrix = pd.DataFrame(
        x_train_std[fn_indices,],
        index=x_train.index[fn_indices],
        columns=x_train.columns,
    )
    weights_labeled = pd.Series(np.ndarray.flatten(model.coef_), index=x_train.columns)
    if ft_contrib_plot_fname:
        plot_overall_feature_contributions(fn_feature_matrix, weights_labeled, ft_contrib_plot_fname)

    return accuracy, precision, recall, f1

def main(percentile=25):
    data = util.impute_by_col(data_loader.load_all_data(), np.mean)
    rates = data.pop('heart_disease_mortality')
    rates_high_low = util.continuous_to_categorical(rates, 100 - percentile)
    train_validation_partitions, test_set = util.split_train_valid_k_fold(data, rates_high_low)

    # Evaluate k fold in parallel and tune hyperparameters
    parameters = []
    metrics = []
    reg_constant_candidates = [0.01, 0.1, 1.0, 10.0, 100]
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for reg_constant in reg_constant_candidates:
        performance_metrics = pool.starmap(log_reg, [part + (reg_constant, "") for i, part in enumerate(train_validation_partitions)])
        overall = np.vstack(performance_metrics)
        logging.info("Average logreg metrics with c={}:\t{}".format(reg_constant, np.mean(overall, axis=0)))
        parameters.append((reg_constant))
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
    
    logging.info("Plotting feature importance for c=0.1")
    list(itertools.starmap(log_reg, [part + (0.1, os.path.join(plotting.PLOTS_DIR, "logreg_kfold_fn_{}.png".format(i))) for i, part in enumerate(train_validation_partitions)]))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

