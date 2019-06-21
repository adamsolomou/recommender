# THIS SHOULD BE IN TENSORFLOW !!

from keras.models import Model
from keras.layers import Embedding, Dense, Flatten
from keras.layers import Activation, Dropout, Input, concatenate
from keras.layers import multiply
from keras import optimizers
import tensorflow as tf
from keras.utils import np_utils
import numpy as np

# unused
weights = tf.constant([1, 1, 1, 1, 1.0])


def custom_loss(yTrue, yPred):
    # scale preds so that the class probas of each sample sum to 1
    output = yPred / tf.reduce_sum(yPred,
                                   reduction_indices=len(yPred.get_shape()) - 1,
                                   keep_dims=True)
    # manual computation of crossentropy
    epsilon = tf.constant(0.00001)
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)

    return - tf.reduce_sum(yTrue * weights * tf.log(output),
                           reduction_indices=len(output.get_shape()) - 1)


values = tf.constant([1, 2, 3, 4, 5], tf.float32)


def my_mse(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=1) + 1

    pred = y_pred * values
    pred = tf.reduce_sum(pred, axis=1)

    y_true = tf.cast(y_true, tf.float32)

    return tf.reduce_mean(tf.square(pred - y_true), axis=0)


def create_model(number_of_users, number_of_movies):
    # define two sets of inputs
    input_movies = Input(shape=(1,), name='input_movies')
    input_users = Input(shape=(1,), name='input_users')

    # the first branch operates on the first input
    embeddings_movies = Embedding(number_of_movies, 100, input_length=1, name='embedding_movies')(input_movies)
    embeddings_movies = Dropout(rate=0.25)(embeddings_movies)

    embeddings_movies = Dense(128, activation='relu', name='dense_251')(embeddings_movies)
    embeddings_movies = Dropout(rate=0.1)(embeddings_movies)
    embeddings_movies = Dense(96, activation='relu', name='dense_253')(embeddings_movies)
    embeddings_movies = Dropout(rate=0.1)(embeddings_movies)

    model_movies = Model(inputs=input_movies, outputs=embeddings_movies)

    # the second branch opreates on the second input
    embeddings_users = Embedding(number_of_users, 150, input_length=1, name='embedding_users')(input_users)
    embeddings_users = Dropout(rate=0.25)(embeddings_users)

    embeddings_users = Dense(256, activation='relu', name='dense_451')(embeddings_users)
    embeddings_users = Dropout(rate=0.1)(embeddings_users)
    embeddings_users = Dense(96, activation='relu', name='dense_452')(embeddings_users)
    embeddings_users = Dropout(rate=0.1)(embeddings_users)

    model_users = Model(inputs=input_users, outputs=embeddings_users)

    # combine the output of the two branches
    combined = concatenate([model_movies.output, model_users.output])

    combined1 = multiply([model_movies.output, model_users.output])
    # combined1 = Activation('tanh')(combined1)

    combined = concatenate([combined, combined1])

    # apply a FC layer and then a regression prediction on the
    # combined outputs
    merged = Flatten()(combined)
    merged = Dense(512, activation='relu', name='dense_511')(merged)
    merged = Dropout(rate=0.1)(merged)
    merged = Dense(5, activation='relu', name='dense_3')(merged)
    merged = Activation('softmax')(merged)

    model = Model(inputs=[model_movies.input, model_users.input], outputs=merged)

    optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.00001)

    model.compile(loss=custom_loss, optimizer=optimizer, metrics=[my_mse])
    return model


class Embeddings:
    def __init__(self,
                 number_of_users,
                 number_of_movies):
        self.number_of_users = number_of_users
        self.number_of_movies = number_of_movies

    def fit_data(self,
                 users_train,
                 movies_train,
                 ratings_train,
                 users_validation=None,
                 movies_validation=None,
                 ratings_validation=None,
                 epochs=40,
                 verbose=True):

        self.model = create_model(self.number_of_movies, self.number_of_movies)

        if verbose:
            self.model.summary()

        validation = False
        if users_validation is not None and movies_validation is not None and ratings_validation is not None:
            validation = True

        users_train = np.array(users_train).reshape(-1, 1)
        movies_train = np.array(movies_train).reshape(-1, 1)

        if validation:
            users_validation = np.array(users_validation).reshape(-1, 1)
            movies_validation = np.array(movies_validation).reshape(-1, 1)

        ratings_categorical_train = np_utils.to_categorical(ratings_train - 1)

        if validation:
            ratings_categorical_validation = np_utils.to_categorical(ratings_validation - 1)

        if validation:
            self.model.fit([movies_train, users_train], ratings_categorical_train, batch_size=512, epochs=epochs,
                           validation_data=([movies_validation, users_validation], ratings_categorical_validation))
        else:
            self.model.fit([movies_train, users_train], ratings_categorical_train, batch_size=512, epochs=epochs)

    def predict(self,
                users_test,
                movies_test):

        users_test = users_test.reshape(-1, 1)
        movies_test = movies_test.reshape(-1, 1)

        predictions = self.model.predict([movies_test, users_test])

        return np.sum(predictions * [1, 2, 3, 4, 5], axis=1)
