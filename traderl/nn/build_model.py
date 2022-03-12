import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GaussianDropout, Conv1D, LayerNormalization

from .block import ConvBlock, ConvnextBlock, FuseBlock, MBBlock
from .layers import Activation, inputs_f, Output, DQNOutput, QRDQNOutput
from .model import *

output_dict = {None: Output, "dqn": DQNOutput, "qrdqn": QRDQNOutput}

gamma = [1, 1.2, 1.4, 1.7, 2., 2.4, 2.9, 3.5]
alpha = [1, 1.1, 1.21, 1.33, 1.46, 1.61, 1.77, 1.94]

l_b0 = (3, 4, 6, 3)

noise = GaussianDropout
noise_b0 = [0.1, 0.1, 0.1]
noise_b1 = [0.1, 0.1, 0.1]
noise_b2 = [0.1, 0.1, 0.1]
noise_b3 = [0.1, 0.1, 0.1]
noise_b4 = [0.1, 0.1, 0.1]
noise_b5 = [0.1, 0.1, 0.1]
noise_b6 = [0.1, 0.1, 0.1]
noise_b7 = [0.1, 0.1, 0.1]
noise_l = [noise_b0, noise_b1, noise_b2, noise_b3, noise_b4, noise_b5, noise_b6, noise_b7]


class BuildModel:
    def __init__(self, num_layer, dim: int, layer_name: str, types: str, scale=0,
                 groups=1, sam=False, se=False, cbam=False, vit=False,
                 efficientv1=False, efficientv2=False,
                 convnext=False):
        self.num_layer = num_layer
        self.layer_name = layer_name.lower()
        self.types = types
        self.groups = groups
        self.sam = bool(sam)
        self.se = bool(se)
        self.cbam = bool(cbam)
        self.efficientv1 = bool(efficientv1)
        self.efficientv2 = bool(efficientv2)
        self.vit = vit
        self.convnext = convnext
        self.attn = "se" if se else "cbam" if cbam else None

        self.noise_ratio = noise_l[scale]
        self.gamma = gamma[scale]
        self.alpha = alpha[scale]
        self.dim = int(dim * self.gamma) if dim else dim
        self.num_layer = num_layer if num_layer else np.round(np.array(l_b0) * self.alpha).astype(int)

        if efficientv1 or efficientv2:
            self.dim = int((16 if self.types == "resnet" else 32) * self.gamma)
        if self.dim and self.layer_name == "lambdalayer":
            self.dim = 4 * int(np.round(self.dim / 4))

    def build_eff_block(self, l):
        block = None
        if self.efficientv1:
            block = [MBBlock for _ in range(len(l))]
        elif self.efficientv2:
            block = [FuseBlock, FuseBlock, FuseBlock]
            block.extend([MBBlock for _ in range(len(l) - 3)])

        self.block = block

    def transition(self, x, dim=None, pool=True):
        if self.types == "densenet":
            dim = x.shape[-1] // 2 if pool else x.shape[-1]
        elif self.types == "resnet":
            dim = self.dim = self.dim * 2 if dim is None else dim

        if self.convnext:
            x = LayerNormalization()(x)
            x = Conv1D(dim, 2, 1, "same", kernel_initializer="he_normal")(x)
            if pool:
                x = tf.keras.layers.AvgPool1D()(x)
            x = LayerNormalization()(x)
        else:
            x = Activation()(x)
            x = Conv1D(dim, 1, 1, "same", kernel_initializer="he_normal")(x)
            if pool:
                x = tf.keras.layers.AvgPool1D()(x)

        return x

    def efficient_model(self, x):
        l = [1, 2, 2, 3, 3, 4, 1]
        k = [3, 3, 5, 3, 5, 5, 3]
        pool = [False, False, True, True, True, False, True]
        self.build_eff_block(l)

        if self.types == "resnet":
            ic = [16, 16, 24, 40, 80, 112, 192]
            oc = [16, 24, 40, 80, 112, 192, 320]
            ep = [1, 6, 6, 6, 6, 6, 6]
        else:
            ic = [32 for _ in range(len(l))]
            oc = ic
            ep = [6 for _ in range(len(l))]

        ic = (np.array(ic) * self.gamma).astype(np.int32)
        oc = (np.array(oc) * self.gamma).astype(np.int32)
        l = np.round(np.array(l) * self.alpha).astype(np.int32)

        if self.layer_name == "lambdalayer":
            ic = [int(4 * np.round(ic / 4)) for ic in ic]
            oc = [int(4 * np.round(oc / 4)) for oc in oc]

        for e, (ic, oc, ep, l, k, pool, block) in enumerate(zip(ic, oc, ep, l, k, pool, self.block)):

            if e != 0:
                x = self.transition(x, oc, pool)

            for _ in range(l):
                x = block(ic, oc, ep, k, 0.25, self.layer_name, self.types, noise, self.noise_ratio[1])(x)

        return x

    def conv_model(self, x):

        for i, l in enumerate(self.num_layer):
            if i != 0:
                x = self.transition(x, None, True)

            for _ in range(l):
                if self.convnext:
                    x = ConvnextBlock(self.dim, self.layer_name, self.types, self.attn, noise, self.noise_ratio[1])(x)
                else:
                    x = ConvBlock(self.dim, self.layer_name, self.types, self.groups, True, self.attn, noise,
                                  self.noise_ratio[1])(x)

        return x

    def build_model(self, input_shape, output_size, output_activation, agent=None):
        inputs, x = inputs_f(input_shape, self.dim, 5, 1, False, "same", noise, self.noise_ratio[0])

        if self.efficientv1 or self.efficientv2:
            x = self.efficient_model(x)
        else:
            x = self.conv_model(x)

        x = tf.keras.layers.GlobalAvgPool1D()(x)

        x = output_dict[agent](output_size, output_activation, noise, self.noise_ratio[2])(x)

        return SAMModel(inputs, x) if self.sam else Model(inputs, x)


network_dict = {}
available_network = []


def create_network(name, num_layer, dim, layer_name, types, **kwargs):
    available_network.append(f"{name}")
    for i in range(8):
        network_dict.update({f"{name}_b{i}": lambda i=i: BuildModel(num_layer, dim, layer_name, types, i, **kwargs)})


create_network("efficientnet", [], 0, "DepthwiseConv1D", "resnet", efficientv1=True)
create_network("sam_efficientnet", [], 0, "DepthwiseConv1D", "resnet", efficientv1=True, sam=True)

create_network("dense_efficientnet", [], 0, "DepthwiseConv1D", "densenet", efficientv1=True)
create_network("sam_dense_efficientnet", [], 0, "DepthwiseConv1D", "densenet", efficientv1=True, sam=True)

create_network("lambda_efficientnet", [], 0, "lambdalayer", "resnet", efficientv1=True)
create_network("sam_lambda_efficientnet", [], 0, "lambdalayer", "resnet", efficientv1=True, sam=True)

create_network("efficientnetv2", [], 0, "DepthwiseConv1D", "resnet", efficientv2=True)
create_network("sam_efficientnetv2", [], 0, "DepthwiseConv1D", "resnet", efficientv2=True, sam=True)

create_network("resnet", [], 48, "Conv1D", "resnet")
create_network("sam_resnet", [], 48, "Conv1D", "resnet", sam=True)
create_network("se_resnet", [], 48, "Conv1D", "resnet", se=True)
create_network("sam_se_resnet", [], 48, "Conv1D", "resnet", se=True, sam=True)

create_network("densenet", [], 48, "Conv1D", "densenet")
create_network("sam_densenet", [], 48, "Conv1D", "densenet", sam=True)
create_network("se_densenet", [], 48, "Conv1D", "densenet", se=True)
create_network("sam_se_densenet", [], 48, "Conv1D", "densenet", se=True, sam=True)

create_network("lambda_resnet", [], 48, "LambdaLayer", "resnet")
create_network("sam_lambda_resnet", [], 48, "LambdaLayer", "resnet", sam=True)
create_network("se_lambda_resnet", [], 48, "LambdaLayer", "resnet", se=True)
create_network("sam_se_lambda_resnet", [], 48, "LambdaLayer", "resnet", se=True, sam=True)

create_network("convnext", [], 48, "DepthwiseConv1D", "resnet", convnext=True)
create_network("sam_convnext", [], 48, "DepthwiseConv1D", "resnet", convnext=True, sam=True)
create_network("se_convnext", [], 48, "DepthwiseConv1D", "resnet", convnext=True, se=True)
create_network("sam_se_convnext", [], 48, "DepthwiseConv1D", "resnet", convnext=True, se=True, sam=True)

create_network("lambda_convnext", [], 48, "LambdaLayer", "resnet", convnext=True)
create_network("sam_lambda_convnext", [], 48, "LambdaLayer", "resnet", convnext=True, sam=True)
create_network("se_lambda_convnext", [], 48, "LambdaLayer", "resnet", convnext=True, se=True)
create_network("sam_se_lambda_convnext", [], 48, "LambdaLayer", "resnet", convnext=True, se=True, sam=True)


def build_model(model_name: str, input_shape: tuple, output_size: int, output_activation=None, agent=None) -> Model:
    model = network_dict[model_name]()
    model = model.build_model(input_shape, output_size, output_activation, agent)

    return model


available_network = np.array(available_network).reshape((-1,))

__all__ = ["build_model", "available_network", "network_dict", "BuildModel"]
