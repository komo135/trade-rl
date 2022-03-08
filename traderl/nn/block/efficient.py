import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from traderl.nn.layers import SE, Activation, layer


class MBBlock(layers.Layer):
    def __init__(self, idim, odim, expand_ratio, kernel_size,
                 se_ratio=0.25, layer_name="DepthwiseConv1D", types="resnet", noise=layers.Dropout, noise_r=0, **kwargs):
        super(MBBlock, self).__init__()
        self.idim = idim
        self.odim = odim
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.kernel_size = kernel_size
        self.layer_name = layer_name.lower()
        self.types = types.lower()
        self.noise = noise
        self.noise_r = noise_r
        assert self.types == "resnet" or self.types == "densenet"

        self.l = np.array([
            Activation(),
            noise(noise_r),
            layer("conv1d", int(idim * expand_ratio), False, 1, 1),
            Activation(),
            layer(layer_name, int(idim * expand_ratio), False, kernel_size, 1),
            Activation(),
            SE(int(idim * expand_ratio), se_ratio) if se_ratio > 0 else None,
            noise(noise_r),
            layer("conv1d", odim, False, kernel_size, 1)
        ])
        self.l = list(self.l[self.l is not None].reshape((-1,)))

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for l in self.l:
            x = l(x)

        if self.types == "resnet":
            return x + inputs
        elif self.types == "densenet":
            return tf.concat([inputs, x], axis=-1)

    def get_config(self):
        config = {
            "idim": self.idim,
            "odim": self.odim,
            "expand_ratio": self.expand_ratio,
            "se_ratio": self.se_ratio,
            "kernel_size": self.kernel_size,
            "layer_name": self.layer_name,
            "types": self.types,
            "noise": self.noise,
            "noise_r": self.noise_r
        }
        return config


class FuseBlock(layers.Layer):
    def __init__(self, idim, odim, expand_ratio, kernel_size,
                 se_ratio=0.25, layer_name="DepthwiseConv1D", types="resnet", noise=layers.Dropout, noise_r=0, **kwargs):
        super(FuseBlock, self).__init__()
        self.idim = idim
        self.odim = odim
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.kernel_size = kernel_size
        self.layer_name = layer_name.lower()
        self.types = types.lower()

        self.l = np.array([
            Activation("mish"),
            noise(noise_r),
            layer("conv1d", int(idim * expand_ratio), False, 1, 1),
            SE(int(idim * expand_ratio), se_ratio) if se_ratio > 0 else None,
            Activation("mish"),
            noise(noise_r),
            layer("conv1d", odim, False, 1, 1),
        ])
        self.l = list(self.l[self.l != None].reshape((-1,)))

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for l in self.l:
            x = l(x)

        if self.types == "resnet":
            return x + inputs
        elif self.types == "densenet":
            return tf.concat([inputs, x], axis=-1)

    def get_config(self):
        config = {
            "idim": self.idim,
            "odim": self.odim,
            "expand_ratio": self.expand_ratio,
            "se_ratio": self.se_ratio,
            "kernel_size": self.kernel_size,
            "layer_name": self.layer_name,
            "types": self.types,
            "noise": self.noise,
            "noise_r": self.noise_r
        }
        return config


__all__ = ["MBBlock", "FuseBlock"]