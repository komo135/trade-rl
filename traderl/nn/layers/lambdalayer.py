from tensorflow.keras import layers, Sequential
import tensorflow as tf


class LambdaLayer(layers.Layer):

    def __init__(self, out_dim, heads=4, use_bias=False, u=4, kernel_size=7, **kwargs):
        super(LambdaLayer, self).__init__()

        self.out_dim = out_dim
        k = 16
        self.heads = heads
        self.v = out_dim // heads
        self.u = u
        self.kernel_size = kernel_size
        self.use_bias = use_bias

        self.top_q = tf.keras.layers.Conv1D(k * heads, 1, 1, "same", use_bias=use_bias)
        self.top_k = tf.keras.layers.Conv1D(k * u, 1, 1, "same", use_bias=use_bias)
        self.top_v = tf.keras.layers.Conv1D(self.v * self.u, 1, 1, "same", use_bias=use_bias)

        self.norm_q = tf.keras.layers.LayerNormalization()
        self.norm_v = tf.keras.layers.LayerNormalization()

        self.rearrange_q = Sequential([
            layers.Reshape((-1, heads, k)),
            layers.Permute((2, 3, 1))
        ])
        self.rearrange_k = Sequential([
            layers.Reshape((-1, u, k)),
            layers.Permute((2, 3, 1))
        ])
        self.rearrange_v = Sequential([
            layers.Reshape((-1, u, self.v)),
            layers.Permute((2, 3, 1))
        ])

        self.rearrange_v2 = layers.Permute((2, 3, 1))
        self.rearrange_lp = layers.Permute((1, 3, 2))
        self.rearrange_output = layers.Reshape((-1, out_dim))

        self.pos_conv = tf.keras.layers.Conv2D(k, (1, self.kernel_size), padding="same")

    def call(self, inputs, *args, **kwargs):
        q = self.top_q(inputs)
        k = self.top_k(inputs)
        v = self.top_v(inputs)

        q = self.norm_q(q)
        v = self.norm_v(v)

        q = self.rearrange_q(q)
        k = self.rearrange_k(k)
        v = self.rearrange_v(v)

        k = tf.nn.softmax(k)

        lc = tf.einsum("b u k n, b u v n -> b k v", k, v)
        yc = tf.einsum("b h k n, b k v -> b n h v", q, lc)

        v = self.rearrange_v2(v)
        lp = self.pos_conv(v)
        lp = self.rearrange_lp(lp)
        yp = tf.einsum("b h k n, b v k n -> b n h v", q, lp)

        y = yc + yp
        output = self.rearrange_output(y)

        return output

    def get_config(self):
        config = {
            "out_dim": self.out_dim,
            "heads": self.heads,
            "use_bias": self.use_bias,
            "u": self.u,
            "kernel_size": self.kernel_size
        }

        return config


__all__ = ["LambdaLayer"]