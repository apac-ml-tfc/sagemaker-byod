from __future__ import absolute_import, print_function

import argparse
import json
import os
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

def main(train_data, train_label, test_data, test_label, batch, epochs,max_items, max_users,n_latent_factors, model_dir):

    movie_input = keras.layers.Input(shape=[1],name='Item')
    movie_embedding = keras.layers.Embedding(max_items, n_latent_factors, name='Movie-Embedding')(movie_input)
    movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)

    user_input = keras.layers.Input(shape=[1],name='User')
    user_vec = keras.layers.Flatten(name='FlattenUsers')(keras.layers.Embedding(max_users, n_latent_factors,name='User-Embedding')(user_input))
    prod = keras.layers.dot([movie_vec, user_vec],axes=1,name='DotProduct')
    model = keras.Model([user_input, movie_input], prod)
    model.compile('adam', 'mean_squared_error')
    
    model.summary()
    model.fit([train_data[:,0], train_data[:,1]], train_label, epochs=epochs, verbose=1)
    
    #score = model.evaluate(test_data, test_label, verbose=1)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])
    
    ###### Save Keras model for TensorFlow Serving ############
    print(f"------ save model to {os.path.join(model_dir, 'model/1/')}")

    sess = tf.keras.backend.get_session()
    tf.saved_model.simple_save(
        sess,
        os.path.join(model_dir, "model/1"),
        inputs={ "inputs": model.input },
        outputs={ t.name: t for t in model.outputs },
    )

def load_training_data(base_dir):
    X_train = np.load(os.path.join(base_dir, 'train_X.npy'))
    y_train = np.load(os.path.join(base_dir, 'train_Y.npy'))
    return X_train, y_train

def load_testing_data(base_dir):
    X_test = np.load(os.path.join(base_dir, 'test_X.npy'))
    y_test = np.load(os.path.join(base_dir, 'test_Y.npy'))
    return X_test, y_test
    

if __name__ == '__main__':
    import keras
    #from keras.optimizers import Adam
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--embedding-size', type=float, default=32)
    parser.add_argument('--max-items', type=float, default=944)
    parser.add_argument('--max-users', type=float, default=1683)

    # input data and model directories
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    args, _ = parser.parse_known_args()
    # arg parsing (shown above) goes here
    train_data, train_label = load_training_data(args.train)
    test_data, test_label = load_testing_data(args.test)
    
    hyperparameters = json.loads(os.environ['SM_HPS'])
    print(hyperparameters)
    epochs=int(hyperparameters['epochs'])
    batch=int(hyperparameters['batch-size'])
    lr=float(hyperparameters['learning-rate'])
    n_latent_factors=int(hyperparameters['embedding-size'])
    max_items=int(hyperparameters['max-items'])
    max_users=int(hyperparameters['max-users'])
    
    main(train_data, train_label, test_data, test_label, batch, epochs,max_items, max_users,n_latent_factors,model_dir=args.model_dir)
