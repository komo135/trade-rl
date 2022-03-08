import tensorflow as tf
from tensorflow.keras import layers
from traderl.nn.layers import layer, Activation
import numpy as np


class ConvBlock(layers.Layer):
    def __init__(self, dim, layer_name="Conv1D", types="resnet",
                 groups=1, bias=True, attention=None, noise=layers.Dropout, noise_r=0, **kwargs):
        """
        :param dim: output dimention
        :param layer_name: layer name
        :param types: "densenet" or "resnet"
        """
        super(ConvBlock, self).__init__()

        self.dim = dim
        self.layer_name = layer_name
        self.types = types.lower()
        self.groups = groups
        self.bias = bias
        self.attention = attention
        self.noise = noise
        self.noise_r = noise_r

        assert self.types == "densenet" or self.types == "resnet"

        if self.types == "resnet":
            self.l = [
                Activation(),
                noise(noise_r),
                layer(layer_name, dim, bias, 7, groups, **kwargs),
                Activation(),
                layer("conv1d", dim * 4, True, 1, 1, **kwargs),
                Activation(),
                layer(attention, dim * 4, **kwargs),
                noise(noise_r),
                layer("conv1d", dim, True, 1, 1, **kwargs),
            ]
        else:
            self.l = [
                Activation(),
                noise(noise_r),
                layer("conv1d", dim * 4, True, 1, 1, **kwargs),
                Activation(),
                layer(attention, dim * 4, **kwargs),
                noise(noise_r),
                layer(layer_name, dim, bias, 7, groups),
            ]

        self.l = np.array(self.l)
        self.l = list(self.l[self.l != None].reshape((-1,)))

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for l in self.l:
            x = l(x)

        if self.types == "densenet":
            return tf.concat([inputs, x], axis=-1)
        elif self.types == "resnet":
            return tf.add(inputs, x)

    def get_config(self):
        config = {
            "dim": self.dim,
            "layer_name": self.layer_name,
            "types": self.types,
            "groups": self.groups,
            "bias": self.bias,
            "attention": self.attention,
            "noise": self.noise,
            "noise_r": self.noise_r
        }
        return config


__all__ = ["ConvBlock"]
