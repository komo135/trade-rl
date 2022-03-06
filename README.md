# Install
```console
git clone https://github.com/komo135/trade-rl.git
cd trade-rl
python setup.py install
```

# Technologies
| Technologies | version |
| -- | -- |
| python | 3.7 |
| tensorflow | 2.7.0 |
| numpy | 1.21.4 |
| pandas | 1.3.4 |
| ta | 0.7.0 |

# How to run

```python
import traderl

csv_url = "https://raw.githubusercontent.com/komo135/forex-historical-data/main/EURUSD/EURUSDh1.csv" # forex data or stoch data

agent = traderl.agent.DQN(df=csv_url, model_name="efficientnet_b0", lr=1e-4, pip_scale=25, n=3, use_device="cpu", 
                          gamma=0.99, train_spread=0.2, spread=10, risk=0.01)
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

agent.train()
```

# Use custom model
```python
import traderl
from tensorflow.keras import layers, optimizers
from traderl import nn

csv_url = "https://raw.githubusercontent.com/komo135/forex-historical-data/main/EURUSD/EURUSDh1.csv" # forex data or stoch data

agent = traderl.agent.DQN(df=csv_url, model_name=None, lr=1e-4, pip_scale=25, n=3, use_device="cpu", 
                          gamma=0.99, train_spread=0.2, spread=10, risk=0.01)

def custom_model():
  dim = 32
  noise = layers.Dropout
  noise_r = 0.1
  
  inputs, x = inputs_f(self.x.shape[1:], dim, 5, 1, False, "same", noise, noise_r)
  x = nn.ConvBlock(dim, "conv1d", "resnet", 1, True, None, noise, noise_r)(x)
  out = DQNOutput(DQNOutput, 2, None, noise, noise_r)(x)
  
  model = nn.model.Model(inputs, x)
  model.compile(optimizers.Adam(agent.lr, clipnorm=1.), nn.losses.DQNLoss)
  
  return model
```
