import tensorflow as tf
from tqdm import tqdm
import numpy as np
from sklearn.utils import shuffle
import math
import os
import shutil

DEFAULT_LOG_PATH = './model_embeddings'


class Embeddings:
    def __init__(self,
                 number_of_users,
                 number_of_movies,
                 embeddings_size=96,
                 dropout_embeddings=0.2,
                 mlp_layers=[1024],
                 dropout=0.1):

        self.number_of_users = number_of_users
        self.number_of_movies = number_of_movies
        self.embeddings_size = embeddings_size
        self.dropout_embeddings = dropout_embeddings
        self.mlp_layers = mlp_layers
        self.dropout = dropout

    def create_graph(self,
                     users,
                     movies,
                     training=True,
                     reuse=False):
        """
        Creates the graph

        Parameters
        ----------
        users       tensor containing the user ids
        movies      tensor containing the movie ids
        training    boolean signifying if we are in train mode or not
        reuse       boolean reuse variables

        Returns
        -------
        tensor containing the ratings
        """

        with tf.variable_scope("embeddings_model", reuse=reuse):
            with tf.name_scope("embeddings"):
                user_embeddings_layers = tf.get_variable('user_embedding',
                                                         [self.number_of_users, self.embeddings_size])

                user_embeddings = tf.nn.embedding_lookup(user_embeddings_layers, users, name='users_lookup')
                user_embeddings = tf.layers.dropout(user_embeddings, rate=self.dropout_embeddings, training=training)

                user_embeddings = tf.layers.dense(user_embeddings, 256, use_bias=True, name='dense_layers_users_1',
                                                  activation=tf.nn.relu)
                user_embeddings = tf.layers.dropout(user_embeddings, rate=self.dropout, training=training)
                user_embeddings = tf.layers.dense(user_embeddings, self.embeddings_size, use_bias=True,
                                                  name='dense_layers_users_2', activation=tf.nn.relu)
                user_embeddings = tf.layers.dropout(user_embeddings, rate=self.dropout, training=training)

                movie_embeddings_layers = tf.get_variable('movie_embedding',
                                                          [self.number_of_movies, self.embeddings_size])

                movie_embeddings = tf.nn.embedding_lookup(movie_embeddings_layers, movies, name='users_lookup')
                movie_embeddings = tf.layers.dropout(movie_embeddings, rate=self.dropout_embeddings, training=training)

                movie_embeddings = tf.layers.dense(movie_embeddings, 256, use_bias=True, name='dense_layers_movies_1',
                                                   activation=tf.nn.relu)
                movie_embeddings = tf.layers.dropout(movie_embeddings, rate=self.dropout, training=training)
                movie_embeddings = tf.layers.dense(movie_embeddings, self.embeddings_size, use_bias=True,
                                                   name='dense_layers_movies_2', activation=tf.nn.relu)
                movie_embeddings = tf.layers.dropout(movie_embeddings, rate=self.dropout, training=training)

            with tf.name_scope("concatenate"):
                multiplied = user_embeddings * movie_embeddings

                merged = tf.concat([user_embeddings, movie_embeddings, multiplied], axis=1)

            with tf.name_scope("feed_forward"):
                for layer in self.mlp_layers:
                    merged = tf.layers.dense(merged, layer, use_bias=True, name='dense_layers_1', activation=tf.nn.relu)
                    merged = tf.layers.dropout(merged, rate=self.dropout, training=training)

                rating_logits = tf.layers.dense(merged, 5, use_bias=True, name='dense_output')

            return rating_logits

    @staticmethod
    def create_dataset(phase, batch_size=512):
        users_placeholder = tf.placeholder(tf.int64, [None], name='users_placeholder_%s' % phase)
        movies_placeholder = tf.placeholder(tf.int64, [None], name='movies_placeholder_%s' % phase)
        ratings_placeholder = tf.placeholder(tf.int64, [None], name='ratings_placeholder_%s' % phase)

        dataset = tf.data.Dataset.from_tensor_slices((users_placeholder, movies_placeholder, ratings_placeholder))

        dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)

        iterator = dataset.make_initializable_iterator()

        return users_placeholder, movies_placeholder, ratings_placeholder, iterator

    @staticmethod
    def logits_to_predictions(prediction_logits):
        predictions_softmax = tf.nn.softmax(prediction_logits)
        return tf.reduce_sum(predictions_softmax * tf.constant([1, 2, 3, 4, 5], dtype=tf.float32), axis=1)

    def square_error(self, true, prediction_logits):

        true = tf.cast(true, tf.float32)

        predictions = self.logits_to_predictions(prediction_logits)

        return tf.reduce_sum((true - predictions) ** 2, name='square_error')

    # noinspection PyUnboundLocalVariable
    def fit_data(self,
                 users_train,
                 movies_train,
                 ratings_train,
                 users_validation=None,
                 movies_validation=None,
                 ratings_validation=None,
                 epochs=40,
                 verbose=True,
                 decay=0.97,
                 decay_steps=2000,
                 learning_rate=1.0,
                 log_path=DEFAULT_LOG_PATH,
                 batch_size=512):

        if os.path.exists(log_path) and os.path.isdir(log_path):
            shutil.rmtree(log_path, ignore_errors=True)

        validation = False
        if users_validation is not None and movies_validation is not None and ratings_validation is not None:
            assert len(users_validation) == len(movies_validation) == len(ratings_validation), \
                "Invalid validation data provided"

            validation = True

        with tf.Graph().as_default():
            with tf.Session() as sess:
                if verbose:
                    print('Creating graph')

                users_train_placeholder, movies_train_placeholder, ratings_train_placeholder, \
                    iterator_train = self.create_dataset(phase='train', batch_size=batch_size)

                users_train_tf, movies_train_tf, ratings_train_tf = iterator_train.get_next()

                ratings_train_predictions = self.create_graph(users_train_tf,
                                                              movies_train_tf, training=True, reuse=False)

                square_error_train = self.square_error(ratings_train_tf, ratings_train_predictions)

                if validation:
                    users_validation_placeholder, movies_validation_placeholder, ratings_validation_placeholder, \
                        iterator_validation = self.create_dataset(phase='validation', batch_size=batch_size)

                    users_validation_tf, movies_validation_tf, ratings_validation_tf = iterator_validation.get_next()

                    ratings_validation_predictions = self.create_graph(users_validation_tf,
                                                                       movies_validation_tf, training=False, reuse=True)

                    square_error_validation = self.square_error(ratings_validation_tf,
                                                                ratings_validation_predictions)

                total_training_loss = tf.nn. \
                    sparse_softmax_cross_entropy_with_logits(logits=ratings_train_predictions,
                                                             labels=ratings_train_tf - 1,
                                                             name="cross_entropy")

                training_loss = tf.reduce_mean(total_training_loss)

                global_step = tf.Variable(1, name='global_step', trainable=False)

                learning_rate = tf.Variable(learning_rate, trainable=False, dtype=tf.float32, name="learning_rate")
                learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay)

                # Gradients and update operation for training the model.
                opt = tf.train.AdadeltaOptimizer(learning_rate)
                #                 opt = tf.train.AdamOptimizer(learning_rate)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                with tf.control_dependencies(update_ops):
                    # Update all the trainable parameters
                    train_op = opt.minimize(training_loss, global_step=global_step)

                saver = tf.train.Saver(max_to_keep=3)

                writer = tf.summary.FileWriter(log_path, sess.graph)
                writer.flush()

                tf.summary.scalar('loss', training_loss)
                tf.summary.scalar('learning_rate', learning_rate)
                tf.summary.scalar('square_error', square_error_train)
                summaries_merged = tf.summary.merge_all()

                sess.run(tf.global_variables_initializer())

                for epoch in range(epochs):

                    users_shuf, movies_shuf, ratings_shuf = shuffle(users_train, movies_train, ratings_train)

                    total_square_error_train = 0
                    total_square_error_validation = 0

                    sess.run(iterator_train.initializer, feed_dict={users_train_placeholder: users_shuf,
                                                                    movies_train_placeholder: movies_shuf,
                                                                    ratings_train_placeholder: ratings_shuf
                                                                    })
                    try:
                        pbar = tqdm(total=len(users_train))
                        pbar.set_description('[Epoch:{:4d}]'.format(epoch + 1))
                        while True:
                            error, summary, _, step = sess.run([square_error_train, summaries_merged,
                                                               train_op, global_step])

                            total_square_error_train += error
                            writer.add_summary(summary, step)

                            pbar.update(batch_size)

                    except tf.errors.OutOfRangeError:
                        pass

                    train_error = math.sqrt(total_square_error_train / len(users_train))

                    saver.save(sess, os.path.join(log_path, "model"), global_step=epoch + 1)

                    if validation:
                        sess.run(iterator_validation.initializer,
                                 feed_dict={users_validation_placeholder: users_validation,
                                            movies_validation_placeholder: movies_validation,
                                            ratings_validation_placeholder: ratings_validation
                                            })

                        try:
                            while True:
                                error = sess.run(square_error_validation)
                                total_square_error_validation += error
                        except tf.errors.OutOfRangeError:
                            pass

                        validation_error = math.sqrt(total_square_error_validation / len(users_validation))
                        pbar.set_postfix_str("Train RMSE: {:3.4f}; Valid RMSE: {:3.4f}"
                                             .format(train_error, validation_error))

                    else:
                        pbar.set_postfix_str("Train RMSE: {:3.4f};".format(train_error))

                    pbar.close()

    def predict(self,
                users_test,
                movies_test,
                log_path=None,
                batch_size=512):

        if log_path is None:
            log_path = DEFAULT_LOG_PATH

        with tf.Graph().as_default():
            with tf.Session() as sess:

                users_placeholder = tf.placeholder(tf.int64, [None], name='users_placeholder')
                movies_placeholder = tf.placeholder(tf.int64, [None], name='movies_placeholder')

                dataset = tf.data.Dataset.from_tensor_slices((users_placeholder, movies_placeholder))
                dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)

                iterator = dataset.make_initializable_iterator()

                users_tf, movies_tf = iterator.get_next()

                ratings_predictions = self.logits_to_predictions(self.create_graph(users_tf,
                                                                                   movies_tf,
                                                                                   training=False,
                                                                                   reuse=False))

                saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint(log_path))

                sess.run(iterator.initializer, feed_dict={users_placeholder: users_test,
                                                          movies_placeholder: movies_test})

                total_predictions = list()

                try:
                    with tqdm(total=len(users_test)) as pbar:
                        while True:
                            predictions_batch = sess.run(ratings_predictions)

                            total_predictions.append(predictions_batch)
                            pbar.update(batch_size)

                except tf.errors.OutOfRangeError:
                    pass

                return np.concatenate(total_predictions, axis=0)
