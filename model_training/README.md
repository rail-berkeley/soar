# Model training code

This directory contains a self-contained python project for training goal-conditioned and language conditioned policies on BridgeData and on Soar-Data.

## Installation
Run in the current directory
```bash
pip install -e .
pip install -r requirements.txt
```

## Structure
- `experiments/`: Contains the main training script `train.py` and the configuration files `train_config.py` and `data_config.py`.
- 'jaxrl_m`: the main library for training models with Jax.
- `jaxrl_m/agents/`: Contains the implementation of the agents.
- `jaxrl_m/data/`: Contains the data processing and data loading code.

## Training
In the current directory, run
```bash
bash experiments/scripts/launch.sh
```
This will launch [train.py](experiments/train.py) with the default arguments specified in [train_config.py](experiments/configs/train_config.py) and [data_config.py](experiments/configs/data_config.py).
