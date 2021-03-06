"""
General utility functions for working with data
"""
import itertools
import logging

import numpy as np
import pandas as pd
import scipy.stats

from sklearn.preprocessing import StandardScaler

def impute_by_col(df, replace=np.nanmedian):
    """Impute nan values in each row using the given replacement"""
    assert isinstance(df, pd.DataFrame)
    for column in df:
        df.at[df[column].isnull(), column] = replace(df[column])
    return df

def isnumeric(x):
    """
    Determines whether the value is numeric. Empty strings and None return False
    Note that while this method can be a bit slower, it is more robust to handling
    cases such as 5e-10 (scientific notation) than other approaches.
    """
    try:
        float(x)
        return True
    except ValueError:
        return False
    except TypeError:
        return False

def continuous_to_categorical(values, percentile_cutoff=75, numeric_cutoff=None):
    """
    Take a 1-dimensional vector of values and return a vector of the same size
    with 0/1 indicating low or high values according to a given percentile cutoff.
    Cutoff can either be percentile or numeric but not both.
    A percentile_cutoff of 75 will give you 1 values for the top 25%, and a
    percentile_cutoff of 90 will give you 1 values for the top 10%
    """
    assert percentile_cutoff is not None or numeric_cutoff is not None, "Must provide either a percentile or numeric cutoff"
    if percentile_cutoff is not None and numeric_cutoff is not None:
        raise NotImplementedError("Cannot provide both percentile and numeric cutoffs")
    x = np.array(values)
    assert x.ndim == 1
    
    if not numeric_cutoff:
        cutoff = np.nanpercentile(x, percentile_cutoff)
        logging.info("{} percentile correspond to a value of {}".format(percentile_cutoff, cutoff))
    else:
        cutoff = numeric_cutoff
        logging.info("Numeric cutoff {} corresponds to approximate percentile {}".format(
            cutoff,
            np.sum(x < cutoff) / len(x)
        ))
    categories = x >= cutoff
    logging.info("Categorized {} of {} values as high".format(np.sum(categories), len(categories)))
    return categories

def split_train_valid_k_fold(full_x, full_y, k=10, testing_holdout=0.1, seed=754927):
    """
    Split the data into k different partitions of training/validation, holding out test data, where each partition
    is returned as a tuple of (x_train, y_train, x_valid, y_valid); these tuples are then returned
    in a list of length. The second return value of this function is a tuple of (x_test, y_test)
    """
    assert 0 <= testing_holdout < 1, "Testing holdout must be a proportion"
    assert full_x.shape[0] == len(full_y)
    np.random.seed(seed)

    # Create a list of indices and shuffle them
    shuf_indices = np.arange(0, len(full_y), dtype=int)
    assert len(shuf_indices) == len(full_y)
    np.random.shuffle(shuf_indices)  # Shuffle in place

    # Create a batch of testing indices that is never include in the train/validation rotation
    testing_holdout_count = int(np.round(testing_holdout * len(full_y)))
    testing_indices = shuf_indices[:testing_holdout_count]  # Takes the first few shuffled indices
    logging.info("Holding out {} indices for testing".format(len(testing_indices)))
    shuf_indices = shuf_indices[testing_holdout_count:]

    partition_size = int(np.ceil(len(shuf_indices) / k))
    partitions = []
    validation_indices_record = []
    for i in range(k):
        lower = partition_size*i
        upper = lower+partition_size
        validation_indices = shuf_indices[lower:upper]
        validation_indices_record.append(validation_indices)
        train_indices = [i for i in shuf_indices if i not in validation_indices]
        assert not set(validation_indices).intersection(set(train_indices))  # Sanity check
        x_valid_sub = full_x.iloc[validation_indices].copy()
        y_valid_sub = full_y[validation_indices]
        x_train_sub = full_x.iloc[train_indices].copy()
        y_train_sub = full_y[train_indices]
        logging.info("K-fold {}: {} testing {} training".format(i, x_valid_sub.shape, x_train_sub.shape))
        partitions.append((x_train_sub, y_train_sub, x_valid_sub, y_valid_sub))
    assert len(partitions) == k  # Make sure we've generated the right number of partitions
    # Make sure we have included ALL the data as test data in one cycle or another
    assert len(set(itertools.chain.from_iterable(validation_indices_record))) == len(full_y) - testing_holdout_count

    # Create the testing dataset
    testing_pair = (full_x.iloc[testing_indices], full_y[testing_indices])

    return partitions, testing_pair

def truncate_extreme_values(df, max_iqr=3):
    """
    For each column in the dataframe, find extreme values according IQR * median and truncate them.
    If max_iqr is set to 3, that means any value that is greater than median + 3 * IQR will be truncated
    and any value less than median - 3 * IQR will be truncated as well

    Written to ignore nan values
    """
    for column in df:
        col_median = np.nanmedian(df[column])
        col_iqr = scipy.stats.iqr(df[column], nan_policy='omit')
        assert col_iqr >= 0, "IQR must be nonnegative but got {} for column {}".format(col_iqr, column)
        if col_iqr == 0:
            logging.info("Skip extreme value truncation for column {}".format(column))
            continue
        max_val = col_median + (max_iqr * col_iqr)
        min_val = col_median - (max_iqr * col_iqr)
        df.at[df[column] > max_val, column] = max_val  # Truncate upper end
        df.at[df[column] < min_val, column] = min_val  # Truncate lower end
    return df

def cross_validate(partitions, model):
    """
    Perform cross validation with the partitions and the model
    Model should already be instantiated and have a fit and predict method.
    Returns two vectors of labels: predictions, truths
    These can easily be fed into sklearn's metrics calculators
    """
    truth_nested = []
    preds_nested = []
    for partition in partitions:
        x_train, y_train, x_test, y_test = partition  # Unpack
        sc = StandardScaler()
        x_train_std = sc.fit_transform(x_train)
        x_test_std = sc.transform(x_test)

        model.fit(x_train_std, y_train)
        preds = model.predict(x_test_std)
        preds_nested.append(preds)
        truth_nested.append(y_test)
    truth_flat = list(itertools.chain.from_iterable(truth_nested))
    preds_flat = list(itertools.chain.from_iterable(preds_nested))
    return truth_flat, preds_flat

def f_beta_score(tp, fp, fn, beta=2):
    """
    All input values should be counts
    The fbeta score measures effectiveness of retrieval WRT a user who attaches BETA times as much importance
    to recall as precision
    https://en.wikipedia.org/wiki/F1_score
    """
    bs = beta * beta
    return ((1 + bs) * tp) / ((1 + bs) * tp + bs * fn + fp)
