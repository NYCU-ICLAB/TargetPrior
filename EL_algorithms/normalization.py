#!/usr/bin/env python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from termcolor import colored

import argparse
import pandas as pd
import numpy as np

def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaler",
                        type=str,
                        choices=["None", "STD", "SCALE"],
                        help='transformer for feature preprocessing')
    parser.add_argument("--train",
                        type=str,
                        help='training set of data')
    parser.add_argument("--test",
                        type=str,
                        help='test set of data')
    return parser.parse_args()

def read_data(data_path):
    token = data_path.split(".")
    name = token[-2].strip('/')
    ext = token[-1]
    df = pd.read_csv(data_path, header=0)

    return df, name

if __name__ == '__main__':
    args = process_command()

    train_df, train_name = read_data(args.train)
    test_df, test_name = read_data(args.test)

    # let me do some simple check for you
    assert (train_df.shape[1] == test_df.shape[1]), \
           "the numbers of feature+label is different between your training and test data!"
    
    train_header = train_df.columns.values
    test_header = test_df.columns.values
    
    for idx in range(len(train_header)):
        assert (train_header[idx] == test_header[idx]), \
            ("the header of column %d is different between your training and test data! \
              It is named to %s in train but %s in test" \
              % (idx, train_header[idx], test_header[idx]))

    
    train_feature = np.array(train_df.iloc[:, 0:-1])
    train_padding = np.expand_dims(np.array(train_df.iloc[:, -1]), axis = 1)
    
    test_feature = np.array(test_df.iloc[:, 0:-1])
    test_padding = np.expand_dims(np.array(test_df.iloc[:, -1]), axis = 1)

    if args.scaler == "STD":
        print(colored("### Using transformer: Standard Scaler", 'green'))
        scaler = StandardScaler()
    elif args.scaler == "SCALE":
        print(colored("### Using transformer: Min Max Scaler", 'green'))
        scaler = MinMaxScaler()
    else:
        assert 0, "No scaler type is provided!"

    scaler.fit(train_feature)
    train_feature = scaler.transform(train_feature)
    test_feature = scaler.transform(test_feature)

    train = np.hstack((train_feature, train_padding)) 
    test = np.hstack((test_feature, test_padding)) 

    pd.DataFrame(train, columns=train_header).to_csv(train_name + "_pre.csv", index=False)
    pd.DataFrame(test, columns=train_header).to_csv(test_name + "_pre.csv", index=False)
