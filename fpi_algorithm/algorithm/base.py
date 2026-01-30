import pickle
from typing import Callable, NamedTuple

import jax

from fpi_algorithm.agent.base import Agent
from fpi_algorithm.utils.experience import Experience


class Algorithm:
    agent: Agent
    alg_state: NamedTuple
    stateless_update: Callable

    def update(self, key: jax.Array, data: Experience, *args) -> dict:
        self.agent.params, self.alg_state, info = self.stateless_update(
            key, self.agent.params, self.alg_state, data, *args)
        return {k: v.item() for k, v in info.items()}

    def save(self, path: str) -> None:
        alg_state = jax.device_get(self.alg_state)
        with open(path, 'wb') as f:
            pickle.dump(alg_state, f)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            alg_state = pickle.load(f)
        self.alg_state = jax.device_put(alg_state)
