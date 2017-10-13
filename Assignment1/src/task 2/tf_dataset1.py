#! usr/bin/python3
'''
Tensorflow dataset 1
'''

import config as cf

import pandas as pd
import tensorflow as tf
import numpy as np


def main():
    # Read in full dataset, without the instances column
    df = pd.read_csv(
        cf.PATHS['SUMDATA_NOISELESS'],
        sep=';',
        nrows=100000
    ).iloc[:, 1:]

    split = int(df.shape[0] * 0.7)
    df_training = df.iloc[:split]
    df_testing = df.iloc[split:]
    print(df_training.shape)
    print(df_testing.shape)
    print(df.shape)


main()
# def linearRegression():


# def supportVectorMachine():


# def kNearestNeighbour():

# def seventyThirtySplit(df):
#    df_training = df
