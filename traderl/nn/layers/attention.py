from tensorflow.keras import layers, Sequential
import tensorflow as tf


class SE(layers.Layer):
    def __init__(self, dim, r=.0625):
        super(SE, self).__init__()
        self.dim = dim
        self.r = r

        self.mlp = Sequential([
            layers.Dense(int(dim * r), "relu", kernel_initializer="he_normal"),
            layers.Dense(dim, "sigmoid")
        ])

    def call(self, inputs, *args, **kwargs):
        x = tf.reduce_mean(inputs, axis=1, keepdims=True)
        x = self.mlp(x)
        x *= inputs

        return x

    def get_config(self):
        config = {
            "dim": self.dim,
            "r": self.r
        }

        return config


class CBAM(layers.Layer):
    def __init__(self, filters, r=.0625):
        super(CBAM, self).__init__()
        self.filters = filters
        self.r = r

        self.avg_pool = layers.GlobalAvgPool1D()
        self.max_pool = layers.GlobalMaxPool1D()
        self.mlp = [
            layers.Dense(int(filters * r), "relu", kernel_initializer="he_normal"),
            layers.Dense(filters)
        ]

        self.concat = layers.Concatenate()
        self.conv = layers.Conv1D(1, 7, 1, "same", activation="sigmoid")

    def compute_mlp(self, x, pool):
        x = pool(x)
        for mlp in self.mlp:
            x = mlp(x)

        return x

    def call(self, inputs, training=None, *args, **kwargs):
        x = self.compute_mlp(inputs, self.avg_pool) + self.compute_mlp(inputs, self.max_pool)
        x = inputs * tf.reshape(tf.nn.sigmoid(x), (-1, 1, self.filters))

        conv = self.concat([tf.reduce_mean(x, -1, keepdims=True), tf.reduce_max(x, -1, keepdims=True)])
        return x * self.conv(conv)

    def get_config(self):
        config = {"filters": self.filters, "r": self.r}
        return config


__all__ = [
    "SE",
    "CBAM"
]