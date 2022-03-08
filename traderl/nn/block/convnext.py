import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from traderl.nn.layers import layer


class ConvnextBlock(layers.Layer):
    def __init__(self, dim: int, layer_name: str, types: str, attention=None, noise=layers.Dropout, noise_r=0,
                 **kwargs):
        super(ConvnextBlock, self).__init__()
        self.dim = dim
        self.layer_name = layer_name
        self.types = types
        self.attention = attention

        self.l = [
            noise(noise_r),
            layer(layer_name, dim, True, 7),
            layers.LayerNormalization(),
            layer("conv1d", dim * 4, True, 1, 1, **kwargs),
            layers.Activation("gelu"),
            layer(attention, dim * 4),
            noise(noise_r),
            layer("conv1d", dim, True, 1, 1, **kwargs),
        ]
        self.l = np.array(self.l)
        self.l = list(self.l[self.l != None].reshape((-1,)))

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for l in self.l:
            x = l(x)

        if self.types == "resnet":
            return inputs + x
        elif self.types == "densenet":
            return tf.concat([inputs, x], -1)

    def get_config(self):
        config = {
            "dim": self.dim,
            "layer_name": self.layer_name,
            "types": self.types,
            "attention": self.attention,
            "noise": self.noise,
            "noise_r": self.noise_r
        }
        return config


__all__ = ["ConvnextBlock"]
