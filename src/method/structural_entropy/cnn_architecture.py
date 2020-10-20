import tensorflow as tf


class ConvNet(tf.keras.Model):
    def __init__(self, parameters):
        super(ConvNet, self).__init__()
        self.parameters = parameters

    def build(self, input_shapes):

        self.conv_1 = tf.keras.layers.Conv1D(filters=self.parameters['num_filters'][0],
                                             kernel_size=(self.parameters['kernel_size'][0]),
                                             data_format='channels_last',
                                             use_bias=True,
                                             activation="relu")
        self.max_pool_1 = tf.keras.layers.MaxPooling1D(pool_size=(2))


        self.conv_2 = tf.keras.layers.Conv1D(filters=self.parameters['num_filters'][1],
                                             kernel_size=(self.parameters['kernel_size'][1]),
                                             data_format='channels_last',
                                             use_bias=True,
                                             activation="relu")
        self.max_pool_2 = tf.keras.layers.MaxPooling1D(pool_size=(2))


        self.conv_3 = tf.keras.layers.Conv1D(filters=self.parameters['num_filters'][2],
                                             kernel_size=(self.parameters['kernel_size'][2]),
                                             data_format='channels_last',
                                             use_bias=True,
                                             activation="relu")

        self.max_pool_3 = tf.keras.layers.MaxPooling1D(pool_size=(2))

        self.drop_1 = tf.keras.layers.Dropout(self.parameters["dropout_rate"])
        self.dense_1 = tf.keras.layers.Dense(self.parameters['hidden'][0],
                                             activation="selu")

        self.drop_2 = tf.keras.layers.Dropout(self.parameters["dropout_rate"])
        self.dense_2 = tf.keras.layers.Dense(self.parameters['hidden'][1],
                                             activation="selu")

        self.drop_3 = tf.keras.layers.Dropout(self.parameters["dropout_rate"])
        self.out = tf.keras.layers.Dense(self.parameters['output'],
                                           activation="softmax")


    def call(self, input_tensor, training=False):
        #print("Input: {}".format(input_tensor.shape))
        input_expanded = tf.keras.backend.expand_dims(input_tensor, axis=-1)
        #print("Input expanded: {}".format(input_expanded.shape))

        conv_1 = self.conv_1(input_expanded)
        #print("Conv 1: {}".format(conv_1.shape))
        pool_1 = self.max_pool_1(conv_1)
        #print("Pool 1: {}".format(pool_1.shape))

        conv_2 = self.conv_2(pool_1)
        #print("Conv 2: {}".format(conv_2.shape))
        pool_2 = self.max_pool_2(conv_2)
        #print("Conv 2: {}".format(pool_2.shape))

        conv_3 = self.conv_3(pool_2)
        #print("Conv 3: {}".format(conv_3.shape))
        pool_3 = self.max_pool_3(conv_3)
        #print("Conv 3: {}".format(pool_3.shape))

        #[-1, int(pool_3.shape[1]) * int(pool_3.shape[2]) * int(pool_3.shape[3])]
        features = tf.reshape(pool_3, shape=[-1, int(pool_3.shape[1]) * int(pool_3.shape[2])])
        #print("Features: {}".format(features.shape))

        drop_1 = self.drop_1(features, training)
        dense_1 = self.dense_1(drop_1)
        #print("Dense 1: {}".format(dense_1.shape))

        drop_2 = self.drop_2(dense_1, training)
        dense_2 = self.dense_2(drop_2)
        #print("Dense 2: {}".format(dense_2.shape))

        drop_3 = self.drop_3(dense_2)
        out = self.out(drop_3)
        #print("Out: {}".format(out.shape))
        return out