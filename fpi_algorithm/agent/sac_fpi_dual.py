import math
from typing import NamedTuple, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.distributions import Normal
from fpi_algorithm.agent.base import Agent
from fpi_algorithm.agent.block import QNet, StochasticPolicyNet


class SACFPIDualParams(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    g1: hk.Params
    g2: hk.Params
    target_g1: hk.Params
    target_g2: hk.Params
    gr1: hk.Params
    gr2: hk.Params
    target_gr1: hk.Params
    target_gr2: hk.Params
    pi: hk.Params
    log_alpha: jax.Array
    log_cg: jax.Array
    lam1: jax.Array
    lam2: jax.Array
    dual_g1: hk.Params
    dual_g2: hk.Params
    dual_target_g1: hk.Params
    dual_target_g2: hk.Params
    dual_pi: hk.Params
    lam3: jax.Array
    lam4: jax.Array


class SACFPIDualAgent(Agent):
    def __init__(
        self,
        key: jax.Array,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int],
        alpha: float = 1.0,
        cg: float = 0.01,
        min_weight: float = 0.1,
        max_weight: float = 10.,
    ):
        def critic_fn(obs, act):
            return QNet(hidden_sizes)(obs, act)

        def scenery_fn(obs, act):
            return QNet(hidden_sizes)(obs, act)

        def actor_fn(obs):
            return StochasticPolicyNet(act_dim, hidden_sizes)(obs)

        critic = hk.without_apply_rng(hk.transform(critic_fn))
        scenery = hk.without_apply_rng(hk.transform(scenery_fn))
        actor = hk.without_apply_rng(hk.transform(actor_fn))

        q1_key, q2_key, g1_key, g2_key, gr1_key, gr2_key, pi_key, \
            dual_g1_key, dual_g2_key, dual_pi_key = jax.random.split(key, 10)
        obs = jnp.zeros((1, obs_dim))
        act = jnp.zeros((1, act_dim))

        q1_params = critic.init(q1_key, obs, act)
        q2_params = critic.init(q2_key, obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        g1_params = scenery.init(g1_key, obs, act)
        g2_params = scenery.init(g2_key, obs, act)
        target_g1_params = g1_params
        target_g2_params = g2_params
        gr1_params = scenery.init(gr1_key, obs, act)
        gr2_params = scenery.init(gr2_key, obs, act)
        target_gr1_params = gr1_params
        target_gr2_params = gr2_params
        pi_params = actor.init(pi_key, obs)
        dual_g1_params = scenery.init(dual_g1_key, obs, act)
        dual_g2_params = scenery.init(dual_g2_key, obs, act)
        dual_target_g1_params = dual_g1_params
        dual_target_g2_params = dual_g2_params
        dual_pi_params = actor.init(dual_pi_key, obs)

        self.params: SACFPIDualParams = SACFPIDualParams(
            q1=q1_params,
            q2=q2_params,
            target_q1=target_q1_params,
            target_q2=target_q2_params,
            g1=g1_params,
            g2=g2_params,
            target_g1=target_g1_params,
            target_g2=target_g2_params,
            gr1=gr1_params,
            gr2=gr2_params,
            target_gr1=target_gr1_params,
            target_gr2=target_gr2_params,
            pi=pi_params,
            log_alpha=jnp.array(math.log(alpha), dtype=jnp.float32),
            log_cg=jnp.array(math.log(cg), dtype=jnp.float32),
            lam1=jnp.array(0, dtype=jnp.float32),
            lam2=jnp.array(0, dtype=jnp.float32),
            dual_g1=dual_g1_params,
            dual_g2=dual_g2_params,
            dual_target_g1=dual_target_g1_params,
            dual_target_g2=dual_target_g2_params,
            dual_pi=dual_pi_params,
            lam3=jnp.array(0, dtype=jnp.float32),
            lam4=jnp.array(0, dtype=jnp.float32),
        )

        self.critic = critic.apply
        self.scenery = scenery.apply
        self.actor = actor.apply
        self.act_dim = act_dim

        def get_single_action(
            key: jax.Array,
            pi_params: hk.Params,
            dual_pi_params: hk.Params,
            obs: jax.Array,
        ):
            mean, std = self.actor(pi_params, obs)
            act = jnp.tanh(Normal(mean, std).sample(key))
            log_weight = jnp.zeros_like(act[..., 0])
            return act, act, log_weight, log_weight

        def get_dual_action(
            key: jax.Array,
            pi_params: hk.Params,
            dual_pi_params: hk.Params,
            obs: jax.Array,
        ):
            mean, std = self.actor(pi_params, obs)
            dist = Normal(mean, std)
            z = dist.sample(key)
            act = jnp.tanh(z)

            dual_mean, dual_std = self.actor(dual_pi_params, obs)
            dual_dist = Normal(dual_mean, dual_std)
            dual_z = dual_dist.sample(key)
            dual_act = jnp.tanh(dual_z)

            logp = dist.log_prob(z).sum(axis=-1)
            dual_logp = dual_dist.log_prob(z).sum(axis=-1)
            logp_dual = dist.log_prob(dual_z).sum(axis=-1)
            dual_logp_dual = dual_dist.log_prob(dual_z).sum(axis=-1)

            log_weight = jnp.clip(dual_logp - logp, math.log(min_weight), math.log(max_weight))
            dual_log_weight = jnp.clip(logp_dual - dual_logp_dual, math.log(min_weight), math.log(max_weight))
            return act, dual_act, log_weight, dual_log_weight

        @jax.jit
        def get_action(
            key: jax.Array,
            pi_params: hk.Params,
            dual_pi_params: hk.Params,
            obs: jax.Array,
            dual: bool,
        ):
            return jax.lax.cond(
                dual,
                get_dual_action,
                get_single_action,
                key, pi_params, dual_pi_params, obs,
            )

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

    def get_action(self, key: jax.Array, obs: np.ndarray, dual: bool) -> np.ndarray:
        res = self._get_action(key, self.params.pi, self.params.dual_pi, jnp.asarray(obs), dual)
        return tuple(np.asarray(elem) for elem in res)

    def get_deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self._get_deterministic_action(self.params.pi, jnp.asarray(obs)))
