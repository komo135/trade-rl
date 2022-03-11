from tensorflow.keras import layers, Sequential
import tensorflow as tf


def inputs_f(input_shape, dim, kernel_size, strides, pooling, padding="same", noise=layers.Dropout, noise_r=0):
    inputs = tf.keras.layers.Input(input_shape)
    x = noise(noise_r)(inputs)
    x = layers.Conv1D(dim, kernel_size, strides, padding, kernel_initializer="he_normal")(x)
    if pooling:
        x = tf.keras.layers.AvgPool1D(3, 2)(x)

    return inputs, x


class Output(layers.Layer):
    def __init__(self, output_size, activation=None, noise=layers.Dropout, noise_r=0):
        super(Output, self).__init__()
        self.output_size = output_size
        self.activation = activation
        self.noise = noise
        self.noise_r = noise_r
        self.out = Sequential([
            noise(noise_r),
            layers.Dense(output_size, activation)
        ])

    def call(self, inputs, *args, **kwargs):
        return self.out(inputs)

    def get_config(self):
        return {"output_size": self.output_size,
                "activation": self.activation,
                "noise": self.noise,
                "noise_r":self.noise_r}


class DQNOutput(Output):
    def __init__(self, output_size, activation=None, noise=layers.Dropout, noise_r=0):
        super(DQNOutput, self).__init__(output_size, activation, noise, noise_r)

        self.out = [
            [noise(noise_r), layers.Dense(output_size), layers.Reshape((output_size, 1))]
            for _ in range(output_size)
        ]

    def call(self, inputs, *args, **kwargs):
        out = []
        for l1 in self.out:
            q = inputs
            for l2 in l1:
                q = l2(q)
            out.append(q)

        return tf.concat(out, axis=-1)


class QRDQNOutput(DQNOutput):
    def __init__(self, output_size, activation=None, noise=layers.Dropout, noise_r=0, quantile_size=32):
        super(QRDQNOutput, self).__init__(output_size, activation, noise, noise_r)
        self.quantile_size = quantile_size

        self.out = [
            [noise(noise_r), layers.Dense(output_size * quantile_size), layers.Reshape((output_size, quantile_size, 1))]
            for _ in range(output_size)
        ]

    def get_config(self):
        config = super(QRDQNOutput, self).get_config()
        config.update({"quantile_size": self.quantile_size})

        return config


__all__ = ["inputs_f", "Output", "DQNOutput", "QRDQNOutput"]