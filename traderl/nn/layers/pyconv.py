from tensorflow.keras import layers


class Pyconv(layers.Layer):
    def __init__(self, dim, groups=32):
        super(Pyconv, self).__init__()

        self.dim = dim
        self.groups = groups

        self.k = k = [3, 5, 7, 9]
        self.conv = [
            layers.Conv1D(dim // 4, k, 1, "same", kernel_initializer="he_normal", groups=groups) for k
            in k
        ]
        self.concat = layers.Concatenate()

    def call(self, inputs, *args, **kwargs):
        x = []
        for conv in self.conv:
            x.append(conv(inputs))

        return self.concat(x)

    def get_config(self):
        config = {
            "dim": self.dim,
            "groups": self.groups
        }

        return config


__all__ = ["Pyconv"]