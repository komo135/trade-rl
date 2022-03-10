import warnings

import numpy as np

from traderl import nn
from traderl.agent import DQN

warnings.simplefilter('ignore')


class QRDQN(DQN):
    agent_name = "QRDQN"
    agent_loss = nn.losses.QRDQNLoss

    def get_actions(self, df):
        q = self.model.predict(df, 10280, workers=10000, use_multiprocessing=True)
        q = np.mean(q, 2)
        actions = np.argmax(q, -1)
        a = np.argmax([q[0, 0, 1], q[0, 1, 0]])
        act = [a]

        for i in range(1, len(actions)):
            a = actions[i, a]
            act.append(a)

        return act

    def target_q(self, returns, target_q, target_a):
        returns = np.reshape(returns, (-1, 2, 1, 2))
        returns = np.tile(returns, (1, 1, 32, 1))

        if self.train_loss and target_q.shape == returns.shape:
            target_a = np.argmax(np.mean(target_a, axis=2), -1)
            rr = range(len(returns))
            returns[:, 0, :, 0] += self.gamma * target_q[rr, 0, :, target_a[:, 0]]
            returns[:, 0, :, 1] += self.gamma * target_q[rr, 1, :, target_a[:, 1]]
            returns[:, 1, :, 0] += self.gamma * target_q[rr, 0, :, target_a[:, 0]]
            returns[:, 1, :, 1] += self.gamma * target_q[rr, 1, :, target_a[:, 1]]

        assert np.mean(np.isnan(returns) == False) == 1

        return returns

    def train(self, epoch=50, batch_size=2056):
        super(QRDQN, self).train(epoch, batch_size)


__all__ = ["QRDQN"]