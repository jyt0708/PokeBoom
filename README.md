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