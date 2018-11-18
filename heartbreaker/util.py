"""
General utility functions for working with data
"""
import itertools
import logging

import numpy as np
import pandas as pd

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
        logging.info("K-fold {}: {}|{} testing {}|{} training".format(i, np.sum(y_valid_sub), x_valid_sub.shape[0], np.sum(y_train_sub), x_train_sub.shape[0]))
        partitions.append((x_train_sub, y_train_sub, x_valid_sub, y_valid_sub))
    assert len(partitions) == k  # Make sure we've generated the right number of partitions
    # Make sure we have included ALL the data as test data in one cycle or another
    assert len(set(itertools.chain.from_iterable(validation_indices_record))) == len(full_y) - testing_holdout_count

    # Create the testing dataset
    testing_pair = (full_x.iloc[testing_indices], full_y[testing_indices])

    return partitions, testing_pair
