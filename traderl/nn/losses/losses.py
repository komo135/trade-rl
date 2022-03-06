from tensorflow.keras.losses import Loss
import tensorflow as tf
import numpy as np


class DQNLoss(Loss):
    k = 2

    def call(self, q_backup, q):
        k = self.k

        error = q_backup - q
        loss = tf.where(tf.abs(error) <= k, error ** 2 * 0.5, 0.5 * k ** 2 + k * (tf.abs(error) - k))
        loss = tf.reduce_mean(loss)

        return loss


class QRDQNLoss(Loss):
    k = 1
    plus_tau = np.arange(1, 33, dtype=np.float32) / 33
    plus_tau = np.reshape(plus_tau, (1, 1, 32, 1))
    minus_tau = np.abs(plus_tau - 1)

    def call(self, q_backup, q):
        k = self.k

        error = q_backup - q
        loss = tf.where(tf.abs(error) <= k, error ** 2 * 0.5, 0.5 * k ** 2 + k * (tf.abs(error) - k))
        loss = tf.where(error > 0, loss * self.plus_tau, loss * self.minus_tau)
        loss = tf.reduce_mean(loss, (0, 1, 3))
        loss = tf.reduce_sum(loss)

        return loss


__all__ = ["DQNLoss", "QRDQNLoss"]