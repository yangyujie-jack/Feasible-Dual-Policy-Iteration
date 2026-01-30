# Feasible-Dual-Policy-Iteration
Code of the paper "Breaking Safety Paradox with Feasible Dual Policy Iteration".
[paper](https://openreview.net/forum?id=BHSSV1nHvU)

## Installation
```bash
# create conda environment
conda create -n fpi python=3.10
conda activate fpi

# install safety-gymnasium
git clone git@git.tsinghua.edu.cn:yangyj21/safety-gymnasium.git
cd safety-gymnasium
pip install -e .

# install dependencies
pip install -U numpy==1.26.4 "jax[cuda12]" dm-haiku optax numpyro tqdm tensorboard tensorboardX matplotlib pandas seaborn

# install fpi-algorithm
pip install -e .
```

## Usage
The training script is `script/train.py`, where the environment, algorithm, and hyperparameters can be configured through command line arguments.
Example:

```bash
python script/train.py --env SafetyPointGoal1-v0 --alg SACFPIDual
```

Training logs, including tensorboard logs and model parameters, are automatically saved under the `log/` folder.

The evaluation and visualization script is `script/plot.py`, which uses tools implemented in `fpi_algorithm/utils/plot.py` including log data collection, training curve plotting, result table building, etc.
The results are saved in `result/` and `figure/` folders.
