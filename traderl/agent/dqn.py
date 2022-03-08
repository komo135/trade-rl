import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from IPython.display import clear_output
from traderl import nn
import ta

warnings.simplefilter('ignore')


class DQN:
    agent_name = "dqn"
    loss = nn.losses.DQNLoss

    def __init__(self, df: pd.DataFrame, model_name, lr=1e-4, pip_scale=25, n=3, use_device="cpu",
                 gamma=0.99, train_spread=0.2, spread=10, risk=0.01):
        """
        :param df: pandas dataframe or csv file. Must contain open, low, high, close
        :param lr: learning rate
        :param model_name: None or model name, If None -> model is not created.
        :param pip_scale: Controls the degree of overfitting
        :param n: int
        :param use_device: tpu or gpu or cpu
        :param gamma: float
        :param train_spread: Determine the degree of long-term training. The smaller the value, the more short-term the trade.
        :param spread: Cost of Trade
        :param risk: What percentage of the balance is at risk
        """

        self.df = df
        self.model_name = model_name
        self.lr = lr
        self.pip_scale = pip_scale
        self.n = n
        self.use_device = use_device.lower()
        self.gamma = gamma
        self.train_spread = train_spread
        self.spread = spread
        self.risk = risk

        self.actions = {0: 1, 1: -1}

        if self.use_device == "tpu":
            try:
                resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
                tf.config.experimental_connect_to_cluster(resolver)
                tf.tpu.experimental.initialize_tpu_system(resolver)
                self.strategy = tf.distribute.TPUStrategy(resolver)
            except:
                self.use_device = "cpu"

        self.train_rewards, self.test_rewards = [], []
        self.max_profits, self.max_pips = [], []
        self.train_loss, self.val_loss = [], []
        self.test_pip, self.test_profit = [], []

        self.x, self.y, self.atr = self.env()
        self.train_step = np.arange(int(len(self.y) * 0.9))
        self.test_step = np.arange(self.train_step[-1], len(self.y))
        self.ind = self.train_step.copy()
        np.random.shuffle(self.ind)

        if self.model_name:
            self.model, self.target_model = self.build_model()

        self.states, self.new_states, self.returns = self.train_data()

        self.max_profit, self.max_pip = 0, 0
        self.now_max_profit, self.now_max_pip = 0, 0

        self.account_size = 100000  # japanese en

    def env(self):
        if isinstance(self.df, str):
            self.df = pd.read_csv(self.df)

        self.df.columns = self.df.columns.str.lower()

        self.df["sig"] = self.df.close - self.df.close.shift(1)
        self.df["atr"] = ta.volatility.average_true_range(self.df.high, self.df.low, self.df.close)
        self.df = self.df.dropna()

        x = []
        y = []
        atr = []

        window_size = 30
        for i in range(window_size, len(self.df.close)):
            x.append(self.df.sig[i - window_size:i])
            y.append(self.df.close[i - 1])
            atr.append(self.df.atr[i - 1])

        x = np.array(x, np.float32).reshape((-1, 30, 1))
        y = np.array(y, np.int32).reshape((-1,))
        atr = np.array(atr, np.int32).reshape((-1,))

        return x, y, atr

    def _build_model(self) -> nn.model.Model:
        model = nn.build_model(self.model_name, self.x.shape[1:], 2, None, self.agent_name)
        model.compile(
            tf.keras.optimizers.Adam(self.lr, clipnorm=1.), loss=self.loss(), steps_per_execution=100
        )
        return model

    def build_model(self):
        if self.use_device == "tpu":
            with self.strategy.scope():
                model = self._build_model()
                target_model = tf.keras.models.clone_model(model)
        else:
            model = self._build_model()
            target_model = tf.keras.models.clone_model(model)
        target_model.set_weights(model.get_weights())

        return model, target_model

    def train_data(self):
        h, h_ = 0, self.train_step[-1]
        n = self.n

        states = self.x[h:h_ - n].copy()
        close = self.y[h:h_]

        buy = np.array([close[i + n] - close[i] for i in range(len(close) - n)]).reshape((-1,))
        scale = np.quantile(abs(buy), 0.99)
        buy = np.clip(buy / scale, -1, 1) * self.pip_scale
        sell = -buy

        spread = self.train_spread * self.pip_scale

        returns = np.zeros((len(close) - n, 2, 2))
        returns[:, 0, 0] = buy
        returns[:, 0, 1] = sell - spread
        returns[:, 1, 0] = buy - spread
        returns[:, 1, 1] = sell

        new_states = np.roll(states, -n, axis=0)[:-n]
        states = states[:-n]
        returns = returns[:-n]

        return states, new_states, returns

    def get_actions(self, df):
        q = self.model.predict(df, 102800, workers=10000, use_multiprocessing=True)
        actions = np.argmax(q, -1)
        a = np.argmax([q[0, 0, 1], q[0, 1, 0]])
        act = [a]

        for i in range(1, len(actions)):
            a = actions[i, a]
            act.append(a)

        return np.array(act).reshape((-1,))

    def trade(self, h, h_):
        df = self.x[h:h_]
        trend = self.y[h:h_]
        atr = self.atr[h:h_]
        profit = self.account_size

        actions = self.get_actions(df)

        old_a = actions[0]
        old_price = trend[0]

        total_pip, total_profit = 0, 0
        self.now_max_profit, self.now_max_pip = 0, 0
        pips, profits = [], []
        total_pips, total_profits = [], []
        buy, sell = [], []

        if old_a == 0:
            buy.append(0)
        else:
            sell.append(0)

        loss_cut = -atr[0] * 2
        position_size = int((profit * self.risk) / -loss_cut)
        position_size = np.minimum(position_size, 500 * 200 * 100)
        position_size = np.maximum(position_size, 1)

        for i, (act, price, atr) in enumerate(zip(actions, trend, atr)):
            if old_a != act:
                old_a = self.actions[old_a]
                pip = (price - old_price) * old_a
                total_pip += pip - self.spread
                pips.append(pip - self.spread)
                total_pips.append(total_pip)

                gain = pip * position_size - self.spread * position_size
                total_profit += gain
                profits.append(gain)
                total_profits.append(total_profit)

                self.now_max_pip = np.maximum(self.now_max_pip, total_pip)
                self.now_max_profit = np.maximum(self.now_max_profit, total_profit)

                old_price = price
                old_a = act

                loss_cut = -atr * 2
                position_size = int((profit * self.risk) / -loss_cut)
                position_size = np.minimum(position_size, 500 * 200 * 100)
                position_size = np.maximum(position_size, 0)

                if act == 0:
                    buy.append(i)
                else:
                    sell.append(i)

        pips = np.array(pips)

        return pips, profits, total_pips, total_profits, total_pip, total_profit, buy, sell

    def evolute(self, h, h_):
        pips, profits, total_pips, total_profits, total_pip, total_profit, buy, sell = self.trade(h, h_)

        acc = np.mean(pips > 0)
        total_win = np.sum(pips[pips > 0])
        total_lose = np.sum(pips[pips < 0])
        rr = total_win / abs(total_lose)
        ev = (np.mean(pips[pips > 0]) * acc + np.mean(pips[pips < 0]) * (1 - acc)) / abs(np.mean(pips[pips < 0]))

        plt.figure(figsize=(10, 5), dpi=100)
        plt.subplot(1, 2, 1)
        plt.plot(total_pips)
        plt.subplot(1, 2, 2)
        plt.plot(total_profits)
        plt.show()

        print(
            f"acc = {acc}, pips = {sum(pips)}\n"
            f"total_win = {total_win}, total_lose = {total_lose}\n"
            f"rr = {rr}, ev = {ev}\n"
        )

    def plot_trade(self, train=False, test=False, period=1):
        assert train or test
        h = 0
        if test:
            h = self.test_step[0]
        elif train:
            h = np.random.randint(0, int(self.train_step[-1] - 960 * period))
        h_ = h + len(self.train_step) // 12 * period
        trend = self.y[h:h_]

        pips, profits, total_pips, total_profits, total_pip, total_profit, buy, sell = self.trade(h, h_)

        plt.figure(figsize=(20, 10), dpi=100)
        plt.plot(trend, color="g", alpha=1, label="close")
        plt.plot(trend, "^", markevery=buy, c="red", label='buy', alpha=0.7)
        plt.plot(trend, "v", markevery=sell, c="blue", label='sell', alpha=0.7)

        plt.legend()
        plt.show()

        print(f"pip = {np.sum(pips)}"
              f"\naccount size = {total_profit}"
              f"\ngrowth rate = {total_profit / self.account_size}"
              f"\naccuracy = {np.mean(np.array(pips) > 0)}")

    def plot_result(self, w, risk=0.1):
        self.model.set_weights(w)
        self.risk = risk

        plt.figure(figsize=(20, 5), dpi=100)
        plt.subplot(1, 2, 1)
        plt.plot(self.test_pip)
        plt.subplot(1, 2, 2)
        plt.plot(self.test_profit)
        plt.show()

        ################################################################################
        self.plot_trade(train=False, test=True, period=9)
        ################################################################################
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss)
        plt.plot(self.val_loss)
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

        len_ = len(self.train_loss) // 2
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss[len_:])
        plt.plot(self.val_loss[len_:])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
        ################################################################################
        self.evolute(self.test_step[0], self.test_step[-1])
        ################################################################################
        plt.figure(figsize=(20, 5), dpi=100)
        plt.subplot(1, 2, 1)
        plt.plot(self.train_rewards)
        plt.subplot(1, 2, 2)
        plt.plot(self.test_rewards)
        plt.show()

        print(f"profits = {self.max_profit}, max profits = {self.max_profits}\n"
              f"pips = {self.max_pip}, max pip = {self.max_pips}")
        ################################################################################
        self.evolute(self.test_step[0] - len(self.train_step), self.test_step[0])

    def target_q(self, returns, target_q, target_a):
        if self.train_loss:
            target_a = np.argmax(target_a, -1)
            rr = range(len(returns))
            returns[:, 0, 0] += self.gamma * target_q[rr, 0, target_a[rr, 0]]
            returns[:, 0, 1] += self.gamma * target_q[rr, 1, target_a[rr, 1]]
            returns[:, 1, 0] += self.gamma * target_q[rr, 0, target_a[rr, 0]]
            returns[:, 1, 1] += self.gamma * target_q[rr, 1, target_a[rr, 1]]

        assert np.mean(np.isnan(returns) == False) == 1

        return returns

    def _train(self, epoch, batch_size):
        ind = self.ind

        states, new_states, returns = self.states[ind].copy(), self.new_states[ind].copy(), self.returns[ind].copy()

        if self.train_loss:
            target_q = self.target_model.predict(new_states, 102800)
        else:
            target_q = np.zeros((len(returns), 2, 2), np.float32)

        for _ in range(epoch):
            returns = self.returns[ind].copy()
            noise = np.random.normal(0, 0.1, states.shape)

            target_a = self.model.predict(new_states + noise, 102800)
            returns = self.target_q(returns, target_q, target_a)

            h = self.model.fit(states + noise, returns, batch_size, validation_split=0.2)
            self.train_loss.extend(h.history["loss"])
            self.val_loss.extend(h.history["val_loss"])

            p = 12

            if len(self.train_loss) >= 200:

                pips, profits, total_pips, total_profits, total_pip, total_profit, buy, sell = \
                    self.trade(self.test_step[0] - len(self.test_step), self.test_step[0])
                self.train_rewards.append(np.sum(pips))
                pips, profits, total_pips, total_profits, total_pip, total_profit, buy, sell = \
                    self.trade(self.test_step[0], self.test_step[-1])
                self.test_rewards.append(np.sum(pips))

                acc = np.mean(pips > 0)

                total_win = np.sum(pips[pips > 0])
                total_lose = np.sum(pips[pips < 0])
                ev = \
                    (np.mean(pips[pips > 0]) * acc + np.mean(pips[pips < 0]) * (1 - acc)) / abs(np.mean(pips[pips < 0]))
                ev = np.clip(ev, 0, 0.75) / 0.75
                rr = np.clip(total_win / abs(total_lose), 0, 2.5) / 2.5
                acc /= 0.7

                self.max_pip = (rr + ev + acc) * np.clip(np.sum(profits) / self.account_size, 1, None)
                self.max_pip = 0 if np.isnan(self.max_pip) else self.max_pip

                self.max_profit /= self.account_size

                self.test_pip.append(self.max_pip)
                self.test_profit.append(self.max_profit)

                if len(pips) >= (p * 5):
                    if self.max_pips <= self.max_pip:
                        self.best_w = self.model.get_weights()
                        self.max_profits = self.max_profit

                    self.max_profits = np.maximum(self.max_profit, self.max_profits)
                    self.max_pips = np.maximum(self.max_pip, self.max_pips)

                plt.figure(figsize=(20, 5), dpi=100)
                plt.subplot(1, 2, 1)
                plt.plot(self.train_rewards)
                plt.subplot(1, 2, 2)
                plt.plot(self.test_rewards)
                plt.show()

                print(f"profits = {self.max_profit}, max profits = {self.max_profits}\n"
                      f"pips = {self.max_pip}, max pip = {self.max_pips}")

    def train(self, epoch=40, batch_size=2056):
        for _ in range(600 // epoch):
            clear_output()
            plt.figure(figsize=(10, 5))
            plt.plot(self.train_loss)
            plt.plot(self.val_loss)
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.show()
            self._train(epoch, batch_size)
            self.target_model.set_weights(self.model.get_weights())


__all__ = ["DQN"]
