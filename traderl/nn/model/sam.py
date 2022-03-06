from tensorflow.keras import Model
import tensorflow as tf


class SAMModel(Model):
    rho = 0.05

    def train_step(self, data):
        x, y = data
        e_ws = []

        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.compiled_loss(y, predictions, regularization_losses=self.losses)

        if "_optimizer" in dir(self.optimizer):  # mixed float policy
            grads_and_vars = self.optimizer._optimizer._compute_gradients(loss, var_list=self.trainable_variables,
                                                                          tape=tape)
        else:
            grads_and_vars = self.optimizer._compute_gradients(loss, var_list=self.trainable_variables, tape=tape)

        grads = [g for g, _ in grads_and_vars]

        grad_norm = self._grad_norm(grads)
        scale = self.rho / (grad_norm + 1e-12)

        for (grad, param) in zip(grads, self.trainable_variables):
            e_w = grad * scale
            e_ws.append(e_w)
            param.assign_add(e_w)

        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.compiled_loss(y, predictions, regularization_losses=self.losses)

        if "_optimizer" in dir(self.optimizer):  # mixed float policy
            grads_and_vars = self.optimizer._optimizer._compute_gradients(loss, var_list=self.trainable_variables,
                                                                          tape=tape)
        else:
            grads_and_vars = self.optimizer._compute_gradients(loss, var_list=self.trainable_variables, tape=tape)
        grads = [g for g, _ in grads_and_vars]

        for e_w, param in zip(e_ws, self.trainable_variables):
            param.assign_sub(e_w)

        grads_and_vars = list(zip(grads, self.trainable_variables))

        self.optimizer.apply_gradients(grads_and_vars)
        self.compiled_metrics.update_state(y, predictions)

        return {m.name: m.result() for m in self.metrics}

    def _grad_norm(self, gradients):
        norm = tf.norm(
            tf.stack([
                tf.norm(grad) for grad in gradients if grad is not None
            ])
        )
        return norm


__all__ = ["SAMModel", "Model"]