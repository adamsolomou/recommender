import tensorflow as tf
import numpy as np
import os
from utils import get_batches
from utils import root_mean_square_error


DEFAULT_LOG_PATH = './autoencoder'


class Autoencoder:
    training = None
    input_pos = None
    input_ = None
    input_mask = None
    intermediate_representation = None
    input_reconstructed = None
    loss = None

    def __init__(self,
                 number_of_users,
                 number_of_movies,
                 activation=tf.nn.relu,
                 layers=None,
                 dropout=None,
                 regularization=0,
                 masking=0.1):

        self.activation = activation

        if layers is None:
            self.layers = [15, 15, 15]
        else:
            self.layers = layers

        self.number_of_users = number_of_users
        self.number_of_movies = number_of_movies

        self.masking = masking
        self.dropout = dropout

        use_regularization = (regularization > 0)
        self.use_regularization = use_regularization

        if regularization == 0:
            # set to small value to avoid tensorflow error
            # use_regularization = False in this case and will not contribute towards the final loss
            self.regularization = 0.1
        else:
            self.regularization = regularization

    def build_graph(self):

        self.training = tf.placeholder(tf.bool, shape=[], name='training')

        self.input_pos = tf.placeholder(tf.int32, shape=[None], name='input_pos')

        # these do not contain the wanted prediction
        self.input_ = tf.placeholder(tf.float32, shape=[None, self.number_of_movies], name='input')

        self.input_mask = tf.placeholder(tf.float32, shape=[None, self.number_of_movies], name='input_mask')

        self.intermediate_representation = self.encode(self.input_)

        self.input_reconstructed = self.decode(self.intermediate_representation, self.input_pos)

        if self.input_mask is not None:
            self.loss = tf.reduce_mean(((self.input_ - self.input_reconstructed) ** 2) * self.input_mask)
        else:
            self.loss = tf.reduce_mean((self.input_ - self.input_reconstructed) ** 2)

        if self.use_regularization:
            self.loss += tf.losses.get_regularization_loss()

    def encode(self,
               input_):

        if self.masking > 0:
            # mask randomly some of the inputs
            input_ = tf.layers.dropout(input_, rate=self.masking, training=self.training)

        x = tf.layers.dense(input_, self.layers[0], use_bias=True, name='input_layer_1_first',
                            activation=self.activation,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization))
        if self.dropout is not None:
            x = tf.layers.dropout(x, rate=self.dropout, training=self.training)

        for i, layer in enumerate(self.layers[1:]):
            x = tf.layers.dense(x, layer, use_bias=True, name='input_layer_1_' + str(i),
                                activation=self.activation,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization))
            if self.dropout is not None:
                x = tf.layers.dropout(x, rate=self.dropout, training=self.training)

        return x

    def decode(self,
               intermediate,
               input_pos):

        x = intermediate
        for i, layer in enumerate(self.layers[::-1][:-1]):
            x = tf.layers.dense(x, layer, use_bias=True, name='input_layer_2_' + str(i),
                                activation=self.activation,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization))
            if self.dropout is not None:
                x = tf.layers.dropout(x, rate=self.dropout, training=self.training)

        bias_input = tf.get_variable(name='bias_input', shape=[self.number_of_users], dtype=tf.float32)
        bias_input_batch = tf.map_fn(lambda y: bias_input[y], input_pos, dtype=tf.float32)

        multiply = tf.constant([self.number_of_movies])

        bias_input_batch_repeated = tf.reshape(tf.tile(bias_input_batch, multiply),
                                               [multiply[0], tf.shape(bias_input_batch)[0]])
        bias_input_batch_repeated = tf.transpose(bias_input_batch_repeated)

        x = tf.layers.dense(x, self.number_of_movies, use_bias=True, name='input_layer_2_final',
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                self.regularization)) + bias_input_batch_repeated

        if self.dropout is not None:
            x = tf.layers.dropout(x, rate=self.dropout, training=self.training)

        return x

    def predict_with_session(self,
                             sess,
                             data,
                             users_test,
                             movies_test):

        # much faster to reconstruct the whole table and make predictions from it
        data_reconstructed = np.zeros((self.number_of_users, self.number_of_movies))

        for users in get_batches(list(range(self.number_of_users)), batch_size=1024, do_shuffle=False):
            user_ratings = [data[i, :] for i in users]

            ratings_reconstructed = sess.run(self.input_reconstructed,
                                             feed_dict={
                                                 self.input_: user_ratings,
                                                 self.input_pos: users,
                                                 self.training: False
                                             })

            data_reconstructed[users] = ratings_reconstructed

        predictions = np.zeros(len(users_test))
        for i, (user, movie) in enumerate(zip(users_test, movies_test)):
            predictions[i] = data_reconstructed[user, movie]

        return predictions

    def predict(self,
                data,
                users_test,
                movies_test,
                log_path=None):

        if log_path is None:
            log_path = DEFAULT_LOG_PATH

        with tf.Graph().as_default():
            with tf.Session() as sess:
                self.build_graph()

                saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint(log_path))

                predictions = self.predict_with_session(sess, data, users_test, movies_test)

                return predictions

    def train(self,
              data,
              data_mask,
              users_validation=None,
              movies_validation=None,
              ratings_validations=None,
              n_epochs=350,
              decay_steps=None,
              learning_rate=None,
              decay=None,
              log_path=None,
              verbose=True):

        if decay_steps is None:
            # empirical
            decay_steps = self.number_of_users // 64 * 5

        if learning_rate is None:
            learning_rate = 10

        if decay is None:
            decay = 0.96

        if log_path is None:
            log_path = DEFAULT_LOG_PATH

        validation = False
        if users_validation is not None and movies_validation is not None and ratings_validations is not None:

            assert len(users_validation) == len(movies_validation) == len(ratings_validations), \
                "Invalid validation data provided"

            validation = True

        with tf.Graph().as_default():
            with tf.Session() as sess:

                self.build_graph()

                global_step = tf.Variable(1, name='global_step', trainable=False)

                learning_rate = tf.Variable(learning_rate, trainable=False, dtype=tf.float32, name="learning_rate")
                learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay)

                # Gradients and update operation for training the model.
                opt = tf.train.AdadeltaOptimizer(learning_rate)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                with tf.control_dependencies(update_ops):
                    # Update all the trainable parameters
                    train_step = opt.minimize(self.loss, global_step=global_step)

                saver = tf.train.Saver(max_to_keep=3)

                writer = tf.summary.FileWriter(log_path, sess.graph)
                writer.flush()

                tf.summary.scalar('loss', self.loss)
                tf.summary.scalar('learning_rate', learning_rate)
                summaries_merged = tf.summary.merge_all()

                sess.run(tf.global_variables_initializer())

                for epoch in range(n_epochs):
                    for users in get_batches(list(range(self.number_of_users))):
                        user_ratings = [data[i, :] for i in users]
                        user_masks = [data_mask[i, :] for i in users]

                        _, loss, step, summary = sess.run([train_step, self.loss, global_step, summaries_merged],
                                                          feed_dict={
                                                              self.input_: user_ratings,
                                                              self.input_pos: users,
                                                              self.input_mask: user_masks,
                                                              self.training: True
                                                          })

                        writer.add_summary(summary, step)

                    writer.flush()

                    if epoch % 10 == 0:
                        if verbose:
                            print('Autoencoder: Done with epoch', epoch)

                            if validation:
                                predictions_validation = self.predict_with_session(sess, data,
                                                                                   users_validation, movies_validation)

                                print('Autoencoder: At epoch', epoch, 'validation error:',
                                      root_mean_square_error(ratings_validations, predictions_validation))

                        saver.save(sess, os.path.join(log_path, "model"), global_step=epoch)
