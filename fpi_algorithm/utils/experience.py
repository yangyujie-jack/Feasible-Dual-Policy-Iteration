from typing import NamedTuple

import numpy as np


class Experience(NamedTuple):
    obs: np.ndarray
    action: np.ndarray
    next_obs: np.ndarray
    reward: np.ndarray
    cost: np.ndarray
    done: np.ndarray


class ExperienceIS(NamedTuple):
    '''
    Experience with importance sampling ratio
    '''
    obs: np.ndarray
    action: np.ndarray
    next_obs: np.ndarray
    reward: np.ndarray
    cost: np.ndarray
    done: np.ndarray
    log_weight: np.ndarray
    log_weight_dual: np.ndarray
