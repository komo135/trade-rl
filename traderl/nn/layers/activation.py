from tensorflow.keras import layers, Sequential
import tensorflow as tf


class Mish(layers.Layer):
    def call(self, inputs, *args, **kwargs):
        return inputs * tf.tanh(tf.math.softplus(inputs))


class Activation(layers.Layer):
    def __init__(self, activation="swish", normalization=layers.LayerNormalization):
        super(Activation, self).__init__()
        self.activation = activation.lower()
        self.normalization = normalization
        if self.activation == "mish":
            self.act = Mish()
        else:
            self.act = layers.Activation(activation)
        if normalization:
            self.norm = normalization()

    def call(self, inputs, training=None, mask=None):
        if self.normalization:
            x = self.norm(inputs)
        else:
            x = inputs
        return self.act(x)

    def get_config(self):
        return {"activation": self.activation, "normalization": self.normalization}


__all__ = [
    "Activation"
]