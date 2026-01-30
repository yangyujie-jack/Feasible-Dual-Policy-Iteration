import math
from typing import NamedTuple, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.distributions import Normal
from fpi_algorithm.agent.base import Agent
from fpi_algorithm.agent.block import DistributionalQNet, StochasticPolicyNet


class DSACTParams(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    pi: hk.Params
    target_pi: hk.Params
    log_alpha: jax.Array


class DSACTAgent(Agent):
    def __init__(
        self,
        key: jax.Array,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int],
        alpha: float = 1.0,
    ): 
        def critic_fn(obs, act):
            return DistributionalQNet(hidden_sizes)(obs, act)

        def actor_fn(obs):
            return StochasticPolicyNet(act_dim, hidden_sizes)(obs)

        critic = hk.without_apply_rng(hk.transform(critic_fn))
        actor = hk.without_apply_rng(hk.transform(actor_fn))

        q1_key, q2_key, pi_key = jax.random.split(key, 3)
        obs = jnp.zeros((1, obs_dim))
        act = jnp.zeros((1, act_dim))
        q1_params = critic.init(q1_key, obs, act)
        q2_params = critic.init(q2_key, obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        pi_params = actor.init(pi_key, obs)
        target_pi_params = pi_params

        self.params: DSACTParams = DSACTParams(
            q1=q1_params,
            q2=q2_params,
            target_q1=target_q1_params,
            target_q2=target_q2_params,
            pi=pi_params,
            target_pi=target_pi_params,
            log_alpha=jnp.array(math.log(alpha), dtype=jnp.float32),
        )

        self.critic = critic.apply
        self.actor = actor.apply
        self.act_dim = act_dim

        @jax.jit
        def get_action(key: jax.Array, pi_params: hk.Params, obs: jax.Array):
            mean, std = self.actor(pi_params, obs)
            return jnp.tanh(Normal(mean, std).sample(key))

        @jax.jit
        def get_deterministic_action(pi_params: hk.Params, obs: jax.Array):
            return jnp.tanh(self.actor(pi_params, obs)[0])

        self._get_action = get_action
        self._get_deterministic_action = get_deterministic_action

    def evaluate(
        self, key: jax.Array, policy_params: hk.Params, obs: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        mean, std = self.actor(policy_params, obs)
        dist = Normal(mean, std)
        z = dist.rsample(key)
        act = jnp.tanh(z)
        logp = (dist.log_prob(z) - 2 * (math.log(2) -
                z - jax.nn.softplus(-2 * z))).sum(axis=-1)
        return act, logp

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self._get_action(key, self.params.pi, jnp.asarray(obs)))

    def get_deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self._get_deterministic_action(self.params.pi, jnp.asarray(obs)))
