from .core import *
from .activation import *
from .attention import *
from .transformer import *
from .pyconv import *
from .skconv import *
from .lambdalayer import *
from .depthwiseconv1d import *
from tensorflow.keras.layers import Conv1D


def layer(layer_name: str, dim, use_bias=True, kernel_size=1, groups=1, **kwargs):
    if layer_name:
        layer_name = layer_name.lower()
        if layer_name == "conv1d":
            return Conv1D(dim, kernel_size, 1, "same", kernel_initializer="he_normal", use_bias=use_bias, groups=groups, **kwargs)
        elif layer_name == "lambdalayer":
            return LambdaLayer(dim, **kwargs)
        elif layer_name == "skconv":
            return SKConv(dim, groups, **kwargs)
        elif layer_name == "depthwiseconv1d":
            return DepthwiseConv1D(kernel_size, 1, "same", kernel_initializer="he_normal", use_bias=use_bias)
        elif layer_name == "se":
            return SE(dim, **kwargs)
        elif layer_name == "cbam":
            return CBAM(dim, **kwargs)
