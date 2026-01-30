from typing import NamedTuple, Sequence

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from fpi_algorithm.agent.base import Agent
from fpi_algorithm.agent.block import QNet, DeterministicPolicyNet 


class TD3Params(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    pi: hk.Params
    target_pi: hk.Params


class TD3Agent(Agent):
    def __init__(
        self,
        key: jax.Array,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int],
        act_noise: float = 0.1,
    ):
        def critic_fn(obs, act):
            return QNet(hidden_sizes)(obs, act)

        def actor_fn(obs):
            return DeterministicPolicyNet(act_dim, hidden_sizes)(obs)

        critic = hk.without_apply_rng(hk.transform(critic_fn))
        actor = hk.without_apply_rng(hk.transform(actor_fn))

        q1_key, q2_key, pi_key = jax.random.split(key, 3)
        obs = jnp.zeros((1, obs_dim))
        act = jnp.zeros((1, act_dim))
        q1_params = critic.init(q1_key, obs, act)
        q2_params = critic.init(q2_key, obs, act)
        pi_params = actor.init(pi_key, obs)
        target_q1_params = q1_params
        target_q2_params = q2_params
        target_pi_params = pi_params

        self.params: TD3Params = TD3Params(
            q1=q1_params,
            q2=q2_params,
            target_q1=target_q1_params,
            target_q2=target_q2_params,
            pi=pi_params,
            target_pi=target_pi_params,
        )

        self.critic = critic.apply 
        self.actor = actor.apply

        @jax.jit
        def get_action(key: jax.Array, pi_params: hk.Params, obs: jax.Array) -> jax.Array:
            act = self.actor(pi_params, obs)
            noise = jax.random.normal(key, shape=act.shape) * act_noise
            return jnp.clip(act + noise, -1.0, 1.0)

        @jax.jit
        def get_deterministic_action(pi_params: hk.Params, obs: jax.Array) -> jax.Array:
            return self.actor(pi_params, obs)

        self._get_action = get_action
        self._get_deterministic_action = get_deterministic_action

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self._get_action(key, self.params.pi, jnp.asarray(obs)))

    def get_deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self._get_deterministic_action(self.params.pi, jnp.asarray(obs)))
