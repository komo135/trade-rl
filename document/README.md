# Agnet
### traderl.agent.DQN
```python
# type Class
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

agent.plot_result()
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

##### 
