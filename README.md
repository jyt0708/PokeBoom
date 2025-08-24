# PokeBoom

## Install PokeBoom

```shell
git clone --recursive git@github.com:jyt0708/PokeBoom.git
```

## Init 

Install Metamon
```shell
cd metamon
pip install -e .
```

Install Server and Run
```shell
cd metamon/server/pokemon-showdown
npm install

node pokemon-showdown start --no-security
```

## Import metamon in our codes

```shell
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "metamon"))
```

# RL Model Input Format

The RL model in this project receives **two types of inputs**: `text` and `numbers`. Both inputs are **2-dimensional tensors**, with the following specifications:

## Shape Convention

| Input      | Shape       | Description                                                                                  |
|------------|------------|----------------------------------------------------------------------------------------------|
| `text`     | `[N, F_text]`   | `N` = number of actions (timesteps), `F_text` = fixed text feature size from observation space |
| `numbers`  | `[N, F_num]`    | `N` = number of actions (timesteps), `F_num` = fixed numeric feature size from observation space |

- **First dimension (`N`)**: Represents the number of actions or timesteps.  
- **Second dimension (`F_text` / `F_num`)**: Fixed number of features determined by the observation space.  
- **Important:** The first dimension of `text` and `numbers` **must be equal** so they can be concatenated in the model.

---

## Online Play / Single Observation

When performing **online inference** or playing a single battle step:

- The first dimension is always `1`, since you only have **one observation at a time**.  
- The shapes are:

```text
text:    [1, F_text]
numbers: [1, F_num]
```

## Examples
- Offline training (sequence of 5 actions):
```text
text.shape    # torch.Size([5, 128])  # 5 timesteps, 128 text features
numbers.shape # torch.Size([5, 48])  # 5 timesteps, 48 numeric features
```

- Online inference (single step):
```text
text.shape    # torch.Size([1, 128])
numbers.shape # torch.Size([1, 48])
```