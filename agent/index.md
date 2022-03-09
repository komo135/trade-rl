# Agent
### traderl.agent.DQN
```python
traderl.agent.DQN(df: pd.DataFrame, model_name, lr=1e-4, pip_scale=25, n=3, use_device="cpu",
                 gamma=0.99, train_spread=0.2, spread=10, risk=0.01)
```
Create DQN agent.

**Example**
```python
import traderl
url = ""
agent = traderl.agent.DQN(url, "efficientnet_b0")
agent.train()

agent.plot_result(agent.best_w)
```

| Args |  |
| -- | -- |
| df | pd.DataFrame or csv file, Must contain open, low, high, close |
| model_name | str or None, If none, the model is not created. See the variable traderl.nn.available_network for available models. |
| lr | float, learning rate |
| pip_scale | int, Scales the reward value.The larger the value, the more intense the overfitting. |
| n | int, Create rewards up to n periods |
| use_device | str, "cpu" or "gpu" or "tpu", Type of device used |
| gamma | float, The larger the value, the more priority is given to rewards in the long term, and the smaller the value, the more priority is given to rewards in the short term. |
| train_spread | float, The cost you impose on a reward when you create it. |
| spread | float, Cost per trade |
| risk | float, Risk per trade |

#### Methods

###### env
```python
env()
```
create environment and returns x, y, atr.

###### _build_model
```python
_build_model()
```
Create a model and return the compiled model.

###### build_model
```python
build_model()
```
Create and return a model and target model.

###### train_data
```python
train_data()
```
Creates data for training and returns states, new_states, and returns.

###### get_actions
```python
get_actions(df)
```
Receives df, generates an action and returns it.

###### trade
```python
trade(h, h_)
```
Execute a trade for the period from h to h_ and return pips, profits, total_pips, total_profits, total_pip, total_profit, buy, sell

###### evolute
```python
evolute(h, h_)
```
Execute a trade for the period from h to h_ and the results of the execution are displayed on the screen.

###### plot_trade
```python
plot_trade(train=False, test=False, period=1)
```
Execute a trade and display the action history on the screen.

###### _train
```python
_train(epoch, batch_size)
```
epoch times, make them learn.

###### train
```
train()
```
Train the model.

###### plot_result
```python
plot_result(w, risk=0.1)
```
Display the training results on the screen.
