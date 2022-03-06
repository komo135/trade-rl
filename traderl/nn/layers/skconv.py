from tensorflow.keras import layers, Sequential
import tensorflow as tf
import numpy as np


class SKConv(layers.Layer):
    def __init__(self, filters: int, groups=32, r=16, **kwargs):
        super(SKConv, self).__init__()
        self.filters = filters
        self.z_filters = np.maximum(filters // r, 32)
        self.groups = groups
        self.r = r

        self.u1 = layers.Conv1D(filters, 3, 1, "same", kernel_initializer="he_normal", groups=groups)
        self.u2 = layers.Conv1D(filters, 5, 1, "same", kernel_initializer="he_normal", dilation_rate=1, groups=groups)

        self.add = layers.Add(axis=-1)

        self.z = layers.Dense(self.z_filters, "elu", kernel_initializer="he_normal")

        self.a = layers.Dense(self.filters)
        self.b = layers.Dense(self.filters)
        self.concat = layers.Concatenate(axis=1)

    def call(self, inputs, *args, **kwargs):
        u1 = self.u1(inputs)
        u2 = self.u2(inputs)

        u = self.add([u1, u2])
        s = tf.reduce_mean(u, axis=1, keepdims=True)
        z = self.z(s)
        a = self.a(z)
        b = self.b(z)
        ab = self.concat([a, b])
        ab = tf.nn.softmax(ab, axis=1)
        a, b = tf.split(ab, 2, 1)

        u1 *= a
        u2 *= b

        return u1 + u2

    def get_config(self):
        config = {
            "filters": self.filters,
            "groups": self.groups,
            "r": self.r
        }

        return config


__all__ = ["SKConv"]