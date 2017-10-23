#! usr/bin/python3
'''
Tensorflow dataset 1
'''

import os

import tempfile
import config as cf
import numpy as np
import pandas as pd
import tensorflow as tf

# Disable TF warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SPLIT_METHOD = cf.FLAGS['TEN_FOLD_CROSS']
#SPLIT_METHOD = cf.FLAGS['SEVENTY_THIRTY']

ALGORITHM = cf.FLAGS['LINEAR_REGRESSION']
#ALGORITHM = ''


def main():
    # Read in full dataset and convert to numpy array
    dataset = pd.read_csv(
        cf.PATHS['SUMDATA_NOISELESS'],
        header=0,
        names=cf.CSV_COLUMNS_SUM,
        sep=';'
    ).values

    # Get rid of Instances column
    dataset = np.delete(dataset, 0, axis=1)
    n_instances = dataset.shape[0]

    # Split the dataset into training and testing arrays
    # Will return lists of size 10 if using 10 fold cross validation
    x_train, x_test, y_train, y_test = split_data_frame(dataset, n_instances)


def split_data_frame(dataset, n_instances):
    x = np.delete(dataset, range(10, 12), axis=1)
    y = np.delete(dataset, range(10), axis=1)
    # If linear regression take the 'Target' columns otherwise take the classification
    # labels
    if ALGORITHM == cf.FLAGS['LINEAR_REGRESSION']:
        y = np.delete(y, 1, axis=1)

    else:
        y = np.delete(y, 0, axis=1)
        y = convert_classification_labels(y)

    if SPLIT_METHOD is cf.FLAGS['SEVENTY_THIRTY']:
        return seventy_thirty(x, y, n_instances)

    elif SPLIT_METHOD is cf.FLAGS['TEN_FOLD_CROSS']:
        return ten_fold_cross(x, y, n_instances)

    else:
        raise Exception('split_data_frame: no split_type set')


def seventy_thirty(x, y, n_instances):
    split = int(n_instances * 0.7)
    x_train = x[0:split]
    x_test = x[split:n_instances]
    y_train = y[0:split]
    y_test = y[split:n_instances]
    print(x_train.shape)
    return x_train, x_test, y_train, y_test


def ten_fold_cross(x, y, n_instances):
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    split = int(n_instances / 10)
    print(n_instances)
    for i in range(10):
        # Note there is a loss of accuracy here converting float to integer
        # The final cross validation set will not reach the final row but
        # each set size is consistent
        test_range = list(range(i * split, (i + 1) * split))
        full_range = list(range(0, n_instances))

        # training_range and test_range are the lists of indices from the whole set of instances
        # to use as the training set for this fold
        training_range = list(set(full_range) - set(test_range))
        x_train.append(x[training_range][:])
        x_test.append(x[test_range][:])
        y_train.append(y[training_range][:])
        y_test.append(y[test_range][:])
    return x_train, x_test, y_train, y_test


def convert_classification_labels(y_data):
    vfunc = np.vectorize(lambda x:
                         {
                             'Very Small Number': 1,
                             'Small Number': 2,
                             'Medium Number': 3,
                             'Large Number': 4,
                             'Very Large Number': 5
                         }[x])
    return vfunc(y_data)


def mean_square_error(y_pred, y_true):
    mse = 0
    n_instances = len(y_pred)
    for pred, truth in zip(y_pred, y_true):
        mse += (pred - truth)**2

    print(mse / n_instances)


main()
