from tensorflow.keras import layers, Sequential
import tensorflow as tf


class PositionAdd(layers.Layer):
    def build(self, input_shape):
        self.pe = self.add_weight("pe", [input_shape[1], input_shape[2]],
                                  initializer=tf.keras.initializers.zeros())

    def call(self, inputs, **kwargs):
        return inputs + self.pe


class MHSA(layers.Layer):
    def __init__(self, dim, num_heads, use_bias=True, noise=layers.Dropout, noise_r=0, **kwargs):
        super(MHSA, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.dim = dim
        self.noise = noise
        self.noise_r = noise_r
        self.use_bias = use_bias

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = layers.Dense(dim * 3, use_bias=use_bias)
        self.qkv_reshape = layers.Reshape((-1, num_heads, head_dim))
        self.qv_permute = layers.Permute((2, 1, 3))
        self.k_permute = layers.Permute((2, 3, 1))
        self.attn_reshape = Sequential([
            layers.Permute((2, 1, 3)),
            layers.Reshape((-1, dim))
        ])
        self.proj = layers.Dense(dim)
        self.drop_out = noise(noise_r)

    def call(self, inputs: tf.Tensor, training=False, *args, **kwargs):
        qkv = self.qkv(inputs)
        q, k, v = tf.split(qkv, 3, -1)

        q = self.qkv_reshape(q)
        k = self.qkv_reshape(k)
        v = self.qkv_reshape(v)

        q = self.qv_permute(q)
        k = self.k_permute(k)
        v = self.qv_permute(v)

        attn = tf.matmul(q, k) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)

        x = tf.matmul(attn, v)
        x = self.attn_reshape(x)
        x = self.proj(x)
        x = self.drop_out(x)

        return x

    def get_config(self):
        config = {
            "num_heads": self.num_heads,
            "dim": self.dim,
            "use_bias": self.use_bias,
            "noise": self.noise,
            "noise_r": self.noise_r
        }
        return config


class TransformerMlp(layers.Layer):
    def __init__(self, dim, mlp_dim, noise=layers.Dropout, noise_r=0):
        super(TransformerMlp, self).__init__()
        self.dim = dim
        self.mlp_dim = mlp_dim
        self.noise = noise
        self.noise_r = noise_r

        self.dense = Sequential([
            layers.Dense(mlp_dim, "gelu"),
            noise(noise_r),
            layers.Dense(dim),
            noise(noise_r)
        ])

    def call(self, inputs, *args, **kwargs):
        return self.dense(inputs)

    def get_config(self):
        config = {
            "dim": self.dim,
            "mlp_dim": self.mlp_dim,
            "noise": self.noise,
            "noise_r": self.noise_r
        }

        return config


class Transformer(layers.Layer):
    def __init__(self, dim, mlp_dim, heads, use_bias=False, noise=layers.Dropout, noise_r=0):
        super(Transformer, self).__init__()
        self.dim = dim
        self.mlp_dim = mlp_dim
        self.heads = heads
        self.use_bias = use_bias

        self.attn = Sequential([layers.LayerNormalization(), MHSA(dim, heads, use_bias, noise, noise_r)])
        self.mlp = Sequential([layers.LayerNormalization(), TransformerMlp(dim, mlp_dim, noise, noise_r)])

    def call(self, inputs, *args, **kwargs):
        x = self.attn(inputs) + inputs
        x = self.mlp(x) + x

        return x

    def get_config(self):
        config = {
            "dim": self.dim,
            "mlp_dim": self.mlp_dim,
            "heads": self.heads,
            "use_bias": self.use_bias
        }

        return config


__all__ = [
    "PositionAdd",
    "MHSA",
    "TransformerMlp",
    "Transformer"
]