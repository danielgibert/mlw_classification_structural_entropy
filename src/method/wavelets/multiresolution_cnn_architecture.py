import tensorflow as tf


class MultiresolutionCNN(tf.keras.Model):
    def __init__(self, parameters):
        super(MultiresolutionCNN, self).__init__()
        self.parameters = parameters

    def build(self, input_shapes):
        self.conv_1 = tf.keras.layers.Conv2D(filters=self.parameters['num_filters'][0],
                                             kernel_size=[2,
                                                          self.parameters['kernel_size'][0]],
                                             data_format='channels_last',
                                             use_bias=True,
                                             activation="relu")
        self.max_pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))

        self.conv_2 = tf.keras.layers.Conv2D(filters=self.parameters['num_filters'][1],
                                             kernel_size=[1, self.parameters['kernel_size'][1]],
                                             data_format='channels_last',
                                             use_bias=True,
                                             activation="relu")
        self.max_pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))

        self.conv_3 = tf.keras.layers.Conv2D(filters=self.parameters['num_filters'][2],
                                             kernel_size=[1, self.parameters['kernel_size'][2]],
                                             data_format='channels_last',
                                             use_bias=True,
                                             activation="relu")
        self.max_pool_3 = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))


        self.drop_1 = tf.keras.layers.Dropout(self.parameters["dropout_rate"])
        self.dense_1 =  tf.keras.layers.Dense(self.parameters['hidden'][0],
                                                           activation="relu")

        self.drop_2 = tf.keras.layers.Dropout(self.parameters["dropout_rate"])
        self.dense_2 = tf.keras.layers.Dense(self.parameters['hidden'][1],
                                                       activation="relu")

        self.drop_3 = tf.keras.layers.Dropout(self.parameters["dropout_rate"])
        self.out = tf.keras.layers.Dense(self.parameters['output'],
                                         activation="softmax")

    def call(self, input_tensor, training=False):
        input_expanded = tf.keras.backend.expand_dims(input_tensor, axis=-1)
        conv_1 = self.conv_1(input_expanded)
        max_pool_1 = self.max_pool_1(conv_1)

        conv_2 = self.conv_2(max_pool_1)
        max_pool_2 = self.max_pool_2(conv_2)

        conv_3 = self.conv_3(max_pool_2)
        max_pool_3 = self.max_pool_3(conv_3)

        features = tf.reshape(max_pool_3, shape=[-1, int(max_pool_3.shape[2]) * int(max_pool_3.shape[3])])

        drop_1 = self.drop_1(features, training)
        dense_1 = self.dense_1(drop_1)

        drop_2 = self.drop_2(dense_1, training)
        dense_2 = self.dense_2(drop_2)

        drop_3 = self.drop_3(dense_2)
        out = self.out(drop_3)
        return out