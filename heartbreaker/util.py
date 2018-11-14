"""
General utility functions for working with data
"""
import itertools

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

def split_test_train_k_fold(full_x, full_y, k=10, seed=754927):
    """
    Split the data into k different partitions of testing and training data, where each partition
    is returned as a tuple of (x_train, y_train, x_test, y_test); these tuples are then returned
    in a list of length k
    """
    assert full_x.shape[0] == len(full_y)
    np.random.seed(seed)
    shuf_indices = np.arange(0, len(full_y), dtype=int)
    assert len(shuf_indices) == len(full_y)
    np.random.shuffle(shuf_indices)  # Shuffle in place
    partition_size = int(np.ceil(len(shuf_indices) / k))
    partitions = []
    test_indices_record = []
    for i in range(k):
        lower = partition_size*i
        upper = lower+partition_size
        test_indices = shuf_indices[lower:upper]
        test_indices_record.append(test_indices)
        train_indices = [i for i in shuf_indices if i not in test_indices]
        assert not set(test_indices).intersection(set(train_indices))  # Sanity check
        x_test_sub = full_x.iloc[test_indices].copy()
        y_test_sub = full_y[test_indices]
        x_train_sub = full_x.iloc[train_indices].copy()
        y_train_sub = full_y[train_indices]
        print("K-fold {}: {}|{} testing {}|{} training".format(i, np.sum(y_test_sub), x_test_sub.shape[0], np.sum(y_train_sub), x_train_sub.shape[0]))
        partitions.append((x_train_sub, y_train_sub, x_test_sub, y_test_sub))
    assert len(partitions) == k  # Make sure we've generated the right number of partitions
    # Make sure we have included ALL the data as test data in one cycle or another
    assert len(set(itertools.chain.from_iterable(test_indices_record))) == len(full_y)
    return partitions
