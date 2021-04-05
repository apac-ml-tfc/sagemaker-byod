import argparse
import joblib
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics



# inference functions ---------------
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf



if __name__ =='__main__':

    print('extracting arguments')
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters
    parser.add_argument('--n-estimators', type=int, default=10)
    parser.add_argument('--min-samples-leaf', type=int, default=3)

    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--train-file', type=str, default='train.csv')
    parser.add_argument('--validation-file', type=str, default='validation.csv')
    parser.add_argument('--features', type=str)  # in this script we ask user to explicitly name features
    parser.add_argument('--target', type=str) # in this script we ask user to explicitly name the target

    args, _ = parser.parse_known_args()

    print('reading data')
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    val_df = pd.read_csv(os.path.join(args.validation, args.validation_file))

    print('building training and testing datasets')
    X_train = train_df[args.features.split()]
    X_val = val_df[args.features.split()]
    y_train = train_df[args.target]
    y_val = val_df[args.target]

    # train
    print('training model')
    #model = RandomForestRegressor(
    #    n_estimators=args.n_estimators,
    #    min_samples_leaf=args.min_samples_leaf,
    #    n_jobs=-1)
    model = DecisionTreeClassifier(criterion="entropy", max_depth=4)
    
    model.fit(X_train, y_train)

    # print abs error
    y_pred = model.predict(X_val)

    print('validating model')
    print("validation:accuracy :",metrics.accuracy_score(y_val, y_pred.astype(int)))
    
    print("validation:auc :",metrics.roc_auc_score(y_val, y_pred))

        
    # persist model
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print('model persisted at ' + path)
    print(args.min_samples_leaf)