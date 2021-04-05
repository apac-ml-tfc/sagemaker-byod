# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Feature engineer and train/test split the customer churn dataset."""

import argparse
import logging
import pathlib

import boto3
import numpy as np
import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split # Import train_test_split function


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.info("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='/opt/ml/processing/input/rawdata.csv')
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"

    logger.info("Reading downloaded data from /opt/ml/processing/input/")
    
    if len(glob.glob(f"{base_dir}/input/*.csv"))>1:
        df = pd.concat(map(pd.read_csv, glob.glob(f"{base_dir}/input/*.csv")))
    else:
        df=pd.read_csv(glob.glob(f"{base_dir}/input/*.csv")[0])
        
    # Drop several columns
    #df = df.drop(['variables'], axis=1)
    # Cast variable
    # df['Area Code'] = df['Area Code'].astype(object)
    # Create dummies
    # model_data = pd.get_dummies(df)
    #model_data = pd.concat([model_data['Churn?_True.'],    #model_data.drop(['Churn?_False.', 'Churn?_True.'], axis=1)], axis=1)
    
    # Split the data
    #split dataset in features and target variable

    #feature_cols = ['Var1', 'Var2', 'Var3', 'Var4']
    #X = df[feature_cols] # Features
    #y = df.Status # Target variable
    
    
   #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 80% training and 20% test
    
    #X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1) # 50% test and 50% val

    
    # Split the data
    train_data, validation_data, test_data = np.split(
        df.sample(frac=1, random_state=1729),
        [int(0.7 * len(df)), int(0.9 * len(df))],
    )
    print('train data size is: ' +str(len(train_data)))
    
    # Create local directories for train/val/test
    
    try:
        os.mkdir(f"{base_dir}/output/train/")
        os.mkdir(f"{base_dir}/output/validation/")
        os.mkdir(f"{base_dir}/output/test/")
        print('Successfully created directories')
    except Exception as e:
    #if the Processing call already creates these directories (or directory otherwise cannot be created)
        print(e)
        print('Could Not Make Directories')
        pass

    # Put csv files in output directories
    pd.DataFrame(train_data).to_csv(f"{base_dir}/output/train/train.csv", header=True, index=False)
    pd.DataFrame(validation_data).to_csv(
        f"{base_dir}/output/validation/validation.csv", header=True, index=False
    )
    pd.DataFrame(test_data).to_csv(
        f"{base_dir}/output/test/test.csv", header=True, index=False
    )
