import os
import sys
import copy
import logging
import itertools
import collections

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch

import data_loader
import util

seed = 754927

class NaiveNet(torch.nn.Dropout):
    """Neural net"""
    def __init__(self, num_features, num_classes=2, first_layer=150, second_layer=25, activation='selu', dropout=0.1):
        super(type(self), self).__init__()
        self.fc1 = torch.nn.Linear(num_features, first_layer, bias=True)
        self.fc2 = torch.nn.Linear(first_layer, second_layer, bias=True)
        self.fc3 = torch.nn.Linear(second_layer, num_classes, bias=True)
        self.activation = activation
        self.p = dropout

    def forward(self, x):
        if self.activation.lower() == 'relu':
            x = torch.nn.functional.leaky_relu(self.fc1(x))
            x = torch.nn.functional.leaky_relu(self.fc2(x))
        elif self.activation.lower() == 'selu':
            x = torch.nn.functional.selu(self.fc1(x))
            x = torch.nn.functional.selu(self.fc2(x))
        else:
            raise ValueError("Unrecognized activation: {}".format(self.activation))
        x = self.fc3(x)
        return torch.nn.functional.softmax(x)

def eval_net_on_test_data(nn, test_data, test_target):
    net_out_test = nn(test_data)
    preds = np.array(net_out_test.data.cpu().max(1)[1])
    preds = np.array(np.round(preds, decimals=0), dtype=int)  # Ensure we are dealing with integer data
    both_pos = np.intersect1d(np.where(test_target.cpu().numpy()), np.where(preds))
    total_positive = np.sum(preds)
    recall = len(both_pos) / sum(test_target.cpu().numpy())
    precision = len(both_pos) / np.sum(preds) if np.sum(preds) else 0.0
    fpr = (total_positive - len(both_pos)) / len(np.where(test_target.cpu().numpy() == 0)[0])  # false positives / total negatives
    return recall, precision, fpr, len(both_pos), (total_positive - len(both_pos)), sum(test_target.cpu().numpy()) - len(both_pos)
    
def train_nn(net, x_train, y_train, x_test, y_test, f_beta=None, weight_ratio=2, weight_decay=0, num_iter=10000, lr=1e-3, seed=6321):
    """
    Trains the neural network, returning the best model based on test dataset performance
    if f_beta is not None, take the model with the highest f-score
    returns dictionaries of precision/recall on test set throughout training iterations
    """
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    # Calculate balanced class weights (originally used [1, 250])
    num_samples = x_train.shape[0]
    bin_count = np.array([sum(y_train == 0), sum(y_train == 1)], dtype=int)
    weights = num_samples / (2 * bin_count)
    weights[1] *= 1 + weight_ratio * weight_ratio  # Upweight the positive cases similar to how f-score weights

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay)  # weight decay is analogous to l2 regularization. 1e-3 seems to be a good value.
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))

    data = torch.autograd.Variable(torch.tensor(x_train, dtype=torch.float32))
    target = torch.autograd.Variable(torch.tensor(y_train))
    test_data = torch.autograd.Variable(torch.tensor(x_test, dtype=torch.float32))
    test_target = torch.autograd.Variable(torch.tensor(y_test))

    tp_values = collections.OrderedDict()
    fp_values = collections.OrderedDict()
    fn_values = collections.OrderedDict()
    precision_values = collections.OrderedDict()
    recall_values = collections.OrderedDict()
    fdr_values = collections.OrderedDict()
    fscore_values = collections.OrderedDict()
    model_history = collections.OrderedDict()

    for i in range(num_iter):
        optimizer.zero_grad()
        net_out = net(data)
        loss = criterion(net_out, target)
        loss.backward()
        optimizer.step()
        if (i + 1) % 1000 == 0:  # Record incrementally
            recall, precision, fpr, tp, fp, fn = eval_net_on_test_data(net, test_data, test_target)
            tp_values[i] = tp
            fp_values[i] = fp
            fn_values[i] = fn
            recall_values[i] = recall
            precision_values[i] = precision
            fdr_values[i] = fpr
            fscore_values[i] = util.f_beta_score(tp, fp, fn, beta=f_beta if f_beta is not None else 1)
            model_history[i] = copy.deepcopy(net)
            logging.info("Training iteration {} - {}".format(i + 1, np.round(fscore_values[i], 6)))

    # Select and return the best model
    best_iteration = None
    iterations = list(precision_values.keys())
    best_iteration = iterations[-1]
    if f_beta is not None:
        f_scores = [fscore_values[i] for i in iterations]
        logging.info("Best F-{} score: {}".format(f_beta, max(f_scores)))
        best_iteration = iterations[np.argmax(f_scores)]
    
    best = model_history[best_iteration]
    logging.info("Returning model at iteration {} with test set precision|recall|Fscore: {}|{}|{}".format(
        best_iteration,
        np.round(precision_values[best_iteration], 4),
        np.round(recall_values[best_iteration], 4),
        np.round(fscore_values[best_iteration], 4),
     ))
    return best, recall_values[best_iteration], precision_values[best_iteration],fscore_values[best_iteration], recall_values, precision_values, fscore_values

def train_nn_simple(net, x_train, y_train, x_test, y_test, weight_ratio=2, weight_decay=0, num_iter=10000, lr=1e-3, seed=6321):
    """
    Trains the neural network, returning the predictions on test set
    """
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    # Calculate balanced class weights (originally used [1, 250])
    num_samples = x_train.shape[0]
    bin_count = np.array([sum(y_train == 0), sum(y_train == 1)], dtype=int)
    weights = num_samples / (2 * bin_count)
    weights[1] *= 1 + weight_ratio * weight_ratio  # Upweight the positive cases similar to how f-score weights

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay)  # weight decay is analogous to l2 regularization. 1e-3 seems to be a good value.
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))

    data = torch.autograd.Variable(torch.tensor(x_train, dtype=torch.float32))
    target = torch.autograd.Variable(torch.tensor(y_train))
    test_data = torch.autograd.Variable(torch.tensor(x_test, dtype=torch.float32))
    test_target = torch.autograd.Variable(torch.tensor(y_test))

    for i in range(num_iter):
        optimizer.zero_grad()
        net_out = net(data)
        loss = criterion(net_out, target)
        loss.backward()
        optimizer.step()
        if (i + 1) % 1000 == 0:  # Record incrementally
            recall, precision, fpr, tp, fp, fn = eval_net_on_test_data(net, test_data, test_target)
    
    # recall, precision, fpr, tp, fp, fn = eval_net_on_test_data(net, test_data, test_target)
    # fscore = util.f_beta_score(tp, fp, fn, beta=1)

    net_out_test = net(test_data)
    preds = np.array(net_out_test.data.cpu().max(1)[1])
    preds = np.array(np.round(preds, decimals=0), dtype=int)

    return preds

def gridsearch():
    """Best parameters seem to be 150 100 0.0001"""
    if torch.cuda.is_available():
        torch.cuda.set_device(2)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
        logging.info(torch.cuda.device_count())
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
        logging.info("CPU")
    
    data = util.impute_by_col(data_loader.load_all_data(), np.mean)
    x = data
    rates = data.pop('heart_disease_mortality')
    y = util.continuous_to_categorical(rates, percentile_cutoff=75)
    
    partitions, holdout = util.split_train_valid_k_fold(x, y)
    
    # Gridsearch for hyperparameters
    first_layer_sizes = [100, 150, 200, 250]
    second_layer_sizes = [25, 50, 75, 100]
    learning_rates = [1e-4, 1e-3, 1e-2]

    param_combos = list(itertools.product(first_layer_sizes, second_layer_sizes, learning_rates))
    f1_scores = []
    for first_layer_size, second_layer_size, lr in param_combos:
        logging.info("Evaluating params: {} {} {}".format(
            first_layer_size,
            second_layer_size,
            lr,
        ))
        all_truths = []
        all_preds = []
        for part in partitions:
            x_train, y_train, x_test, y_test = part
            sc = StandardScaler()
            # Fit the scaler to the training data and transform
            x_train_std = sc.fit_transform(x_train)
            # Apply the scaler to the test data
            x_test_std = sc.transform(x_test)
            preds = train_nn_simple(
                NaiveNet(x_train.shape[1], first_layer=first_layer_size, second_layer=second_layer_size),
                x_train_std, y_train.astype(int), x_test_std, y_test.astype(int),
                weight_ratio=1,
                lr=lr,
            )
            all_truths.extend(y_test)
            all_preds.extend(preds)
        s = f1_score(all_truths, all_preds)
        logging.info("Evaluating params {} {} {} | {}".format(
            np.round(first_layer_size, 4),
            np.round(second_layer_size, 4),
            np.round(lr, 4),
            s
        ))
        f1_scores.append(s)
    best_index = np.argmax(f1_scores)
    logging.info("Best model params: {}".format(param_combos[best_index]))

def eval_params_on_train(first_layer=150, second_layer=100, lr=1e-4):
    if torch.cuda.is_available():
        torch.cuda.set_device(2)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
        logging.info(torch.cuda.device_count())
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
        logging.info("CPU")
    
    data = util.impute_by_col(data_loader.load_all_data(), np.mean)
    x = data
    rates = data.pop('heart_disease_mortality')
    y = util.continuous_to_categorical(rates, percentile_cutoff=75)
    
    partitions, holdout = util.split_train_valid_k_fold(x, y)

    all_truths = []
    all_preds = []
    for part in partitions:
        x_train, y_train, x_test, y_test = part
        sc = StandardScaler()
        # Fit the scaler to the training data and transform
        x_train_std = sc.fit_transform(x_train)
        # Apply the scaler to the test data
        x_test_std = sc.transform(x_test)
        preds = train_nn_simple(
            NaiveNet(x_train.shape[1], first_layer=first_layer, second_layer=second_layer),
            x_train_std, y_train.astype(int), x_test_std, y_test.astype(int),
            weight_ratio=1,
            lr=lr,
        )
        all_truths.extend(y_test)
        all_preds.extend(preds)

    logging.info("Accuracy: {}".format(accuracy_score(all_truths, all_preds)))
    logging.info("Precision: {}".format(precision_score(all_truths, all_preds)))
    logging.info("Recall: {}".format(recall_score(all_truths, all_preds)))
    logging.info("F1 score: {}".format(f1_score(all_truths, all_preds)))

def eval_params_on_test(first_layer=150, second_layer=100, lr=1e-4):
    if torch.cuda.is_available():
        torch.cuda.set_device(2)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
        logging.info(torch.cuda.device_count())
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
        logging.info("CPU")
    
    data = util.impute_by_col(data_loader.load_all_data(), np.mean)
    x = data
    rates = data.pop('heart_disease_mortality')
    y = util.continuous_to_categorical(rates, percentile_cutoff=75)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=seed)

    sc = StandardScaler()
    # Fit the scaler to the training data and transform
    x_train_std = sc.fit_transform(x_train)
    # Apply the scaler to the test data
    x_test_std = sc.transform(x_test)
    preds = train_nn_simple(
        NaiveNet(x_train.shape[1], first_layer=first_layer, second_layer=second_layer),
        x_train_std, y_train.astype(int), x_test_std, y_test.astype(int),
        weight_ratio=1,
        lr=lr,
    )

    logging.info("Test Accuracy: {}".format(accuracy_score(y_test, preds)))
    logging.info("Test Precision: {}".format(precision_score(y_test, preds)))
    logging.info("Test Recall: {}".format(recall_score(y_test, preds)))
    logging.info("Test F1 score: {}".format(f1_score(y_test, preds)))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # gridsearch()
    # eval_params_on_train()
    eval_params_on_test()
