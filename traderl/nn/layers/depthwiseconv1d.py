from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers.convolutional import Conv1D
# imports for backwards namespace compatibility
# pylint: disable=unused-import
# pylint: enable=unused-import
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn


# pylint: disable=g-classes-have-attributes


class DepthwiseConv1D(Conv1D):
    """Depthwise 1D convolution.

    Depthwise convolution is a type of convolution in which a single convolutional
    filter is apply to each input channel (i.e. in a depthwise way).
    You can understand depthwise convolution as being
    the first step in a depthwise separable convolution.

    It is implemented via the following steps:

    - Split the input into individual channels.
    - Convolve each input with the layer's kernel (called a depthwise kernel).
    - Stack the convolved outputs together (along the channels axis).

    Unlike a regular 1D convolution, depthwise convolution does not mix
    information across different input channels.

    The `depth_multiplier` argument controls how many
    output channels are generated per input channel in the depthwise step.

    Args:
      kernel_size: An integer or tuple/list of 2 integers, specifying the
        height and width of the 1D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the height and width.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: one of `'valid'` or `'same'` (case-insensitive).
        `"valid"` means no padding. `"same"` results in padding with zeros evenly
        to the left/right or up/down of the input such that output has the same
        height/width dimension as the input.
      depth_multiplier: The number of depthwise convolution output channels
        for each input channel.
        The total number of depthwise convolution output
        channels will be equal to `filters_in * depth_multiplier`.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch_size, height, width, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch_size, channels, height, width)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be 'channels_last'.
      dilation_rate: An integer or tuple/list of 2 integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied (
        see `keras.activations`).
      use_bias: Boolean, whether the layer uses a bias vector.
      depthwise_initializer: Initializer for the depthwise kernel matrix (
        see `keras.initializers`). If None, the default initializer (
        'glorot_uniform') will be used.
      bias_initializer: Initializer for the bias vector (
        see `keras.initializers`). If None, the default initializer (
        'zeros') will bs used.
      depthwise_regularizer: Regularizer function applied to
        the depthwise kernel matrix (see `keras.regularizers`).
      bias_regularizer: Regularizer function applied to the bias vector (
        see `keras.regularizers`).
      activity_regularizer: Regularizer function applied to
        the output of the layer (its 'activation') (
        see `keras.regularizers`).
      depthwise_constraint: Constraint function applied to
        the depthwise kernel matrix (
        see `keras.constraints`).
      bias_constraint: Constraint function applied to the bias vector (
        see `keras.constraints`).

      Input shape:
        3D tensor with shape:
        `(batch_size, channels, steps)` if data_format='channels_first'
        or 3D tensor with shape:
        `(batch_size, steps, channels)` if data_format='channels_last'.

      Output shape:
        3D tensor with shape:
        `(batch_size, filters, new_steps)` if data_format='channels_first'
        or 3D tensor with shape:
        `(batch_size,  new_steps, filters)` if data_format='channels_last'.
        `new_steps` value might have changed due to padding or strides.

      Returns:
        A tensor of rank 3 representing
        `activation(separableconv1d(inputs, kernel) + bias)`.

    Raises:
      ValueError: when both `strides` > 1 and `dilation_rate` > 1.
    """

    def __init__(self,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 depth_multiplier=1,
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DepthwiseConv1D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        depthwise_kernel_shape = self.kernel_size + (input_dim,
                                                     self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        if self.padding == 'causal':
            inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))
        if self.data_format == 'channels_last':
            strides = (1,) + self.strides * 2 + (1,)
            spatial_start_dim = 1
        else:
            strides = (1, 1) + self.strides * 2
            spatial_start_dim = 2

        # Explicitly broadcast inputs and kernels to 4D.
        inputs = array_ops.expand_dims(inputs, spatial_start_dim)
        depthwise_kernel = array_ops.expand_dims(self.depthwise_kernel, 0)
        dilation_rate = (1,) + self.dilation_rate

        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding
        op_padding = op_padding.upper()

        outputs = nn.depthwise_conv2d(
            inputs,
            depthwise_kernel,
            strides=strides,
            padding=op_padding,
            rate=dilation_rate,
            data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

        if self.use_bias:
            outputs = nn.bias_add(
                outputs,
                self.bias,
                data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

        outputs = array_ops.squeeze(outputs, [spatial_start_dim])

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = super(DepthwiseConv1D, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = initializers.serialize(
            self.depthwise_initializer)
        config['depthwise_regularizer'] = regularizers.serialize(
            self.depthwise_regularizer)
        config['depthwise_constraint'] = constraints.serialize(
            self.depthwise_constraint)
        return config


__all__ = ["DepthwiseConv1D"]