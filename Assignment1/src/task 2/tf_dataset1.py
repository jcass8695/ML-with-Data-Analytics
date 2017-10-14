#! usr/bin/python3
'''
Tensorflow dataset 1
'''

import tempfile
import config as cf

import pandas as pd
import tensorflow as tf
import numpy as np


def main():
    CSV_COLUMNS = ['Instances', 'Feature_1', 'Feature_2', 'Feature_3', 'Feature_4',
                   'Feature_5', 'Feature_6', 'Feature_7', 'Feature_8', 'Feature_9',
                   'Feature_10', 'Target', 'Target_Class']
    # Read in full dataset
    df = pd.read_csv(
        cf.PATHS['SUMDATA_NOISELESS'],
        header=0,
        names=CSV_COLUMNS,
        sep=';'
    )
    # Split the dataset into training and testing
    # Leave out the Instances column
    df_training, df_testing = split_data_frame(df, cf.FLAGS['SEVENTY_THIRTY'])

    # Format the categorical class values into integer representations
    # Note indexing a pandas DataFrame like this returns a Series object
    train_labels_regression = df_training['Target']
    train_label_classification = (df_training['Target_Class']
                                  .apply(lambda x:
                                         {
                                             'Very Small Number': 1,
                                             'Small Number': 2,
                                             'Medium Number': 3,
                                             'Large Number': 4,
                                             'Very Large Number': 5
                                         }[x]
                                         ))

    test_label_regression = df_testing['Target']
    test_label_classification = (df_testing['Target_Class']
                                 .apply(lambda x:
                                        {
                                            'Very Small Number': 1,
                                            'Small Number': 2,
                                            'Medium Number': 3,
                                            'Large Number': 4,
                                            'Very Large Number': 5
                                        }[x]
                                        ))

    # Build TF feature columns
    base_feature_columns = build_feature_columns()

    # Build training and testing input function
    training_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=df_training,
        y=train_labels_regression,
        shuffle=False
    )

    testing_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=df_testing,
        y=test_label_regression,
        shuffle=False
    )

    model = tf.estimator.LinearRegressor(
        model_dir=tempfile.mkdtemp(),
        feature_columns=base_feature_columns
    )

    model.train(
        input_fn=training_input_fn
    )

    results = model.evaluate(input_fn=testing_input_fn)
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))


def split_data_frame(df, split_type):
    if split_type is cf.FLAGS['SEVENTY_THIRTY']:
        split = int(df.shape[0] * 0.7)
        df_training = df.iloc[:split, 1:]
        df_testing = df.iloc[split:, 1:]
        return df_training, df_testing

    elif split_type is cf.FLAGS['TEN_FOLD_CROSS']:
        # TODO: implement 10 fold
        return None, None


def build_feature_columns():
    feature1 = tf.feature_column.numeric_column('Feature_1')
    feature2 = tf.feature_column.numeric_column('Feature_2')
    feature3 = tf.feature_column.numeric_column('Feature_3')
    feature4 = tf.feature_column.numeric_column('Feature_4')
    feature5 = tf.feature_column.numeric_column('Feature_5')
    feature6 = tf.feature_column.numeric_column('Feature_6')
    feature7 = tf.feature_column.numeric_column('Feature_7')
    feature8 = tf.feature_column.numeric_column('Feature_8')
    feature9 = tf.feature_column.numeric_column('Feature_9')
    feature10 = tf.feature_column.numeric_column('Feature_10')
    return [feature1, feature2, feature3, feature4, feature5,
            feature6, feature7, feature8, feature9, feature10]


main()
