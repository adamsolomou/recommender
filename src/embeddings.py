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

def customLoss(yTrue, yPred):
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
    inputMovies = Input(shape=(1,), name='input_movies')
    inputUsers = Input(shape=(1,), name='input_users')

    # the first branch operates on the first input
    embeddings_movies = Embedding(number_of_movies, 100, input_length=1, name='embedding_movies')(inputMovies)
    embeddings_movies = Dropout(rate=0.25)(embeddings_movies)
    model_movies = Model(inputs=inputMovies, outputs=embeddings_movies)

    # the second branch opreates on the second input
    embeddings_users = Embedding(number_of_users, 150, input_length=1, name='embedding_users')(inputUsers)
    embeddings_users = Dropout(rate=0.25)(embeddings_users)
    model_users = Model(inputs=inputUsers, outputs=embeddings_users)

    # combine the output of the two branches
    combined = concatenate([model_movies.output, model_users.output])
    combined = Flatten()(combined)

    choose_movies = Dense(512, activation='relu')(combined)
    choose_movies = Dropout(rate=0.1)(choose_movies)
    choose_movies = Dense(100, activation='sigmoid')(choose_movies)
    choose_movies = Dropout(rate=0.1)(choose_movies)

    #     movies_embeddings = Activation('tanh')(model_movies.output)
    movies_embeddings = model_movies.output
    movies_embeddings = multiply([movies_embeddings, choose_movies])

    choose_users = Dense(512, activation='relu')(combined)
    choose_users = Dropout(rate=0.1)(choose_users)
    choose_users = Dense(150, activation='sigmoid')(choose_users)
    choose_users = Dropout(rate=0.1)(choose_users)

    #     users_embeddings = Activation('tanh')(model_users.output)
    users_embeddings = model_users.output
    users_embeddings = multiply([users_embeddings, choose_users])

    merged = concatenate([movies_embeddings, users_embeddings])

    # apply a FC layer and then a regression prediction on the
    # combined outputs
    merged = Flatten()(merged)
    merged = Dense(512, activation='relu', name='dense_511')(merged)
    merged = Dropout(rate=0.1)(merged)
    merged = Dense(5, activation='relu')(merged)
    merged = Activation('softmax')(merged)

    # our model will accept the inputs of the two branches and
    # then output a single value
    model = Model(inputs=[model_movies.input, model_users.input], outputs=merged)

    optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.00001)
    # optimizer = optimizers.Adadelta(lr=10.0, rho=0.95, epsilon=None, decay=0.0)

    model.compile(loss=customLoss, optimizer=optimizer, metrics=[my_mse])
    #     model.compile(loss='mean_squared_error', optimizer=optimizer)

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
                 epochs=40,
                 users_validation=None,
                 movies_validation=None,
                 ratings_validation=None,
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
