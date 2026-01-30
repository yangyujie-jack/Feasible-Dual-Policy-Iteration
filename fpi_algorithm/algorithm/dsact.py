from typing import NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from fpi_algorithm.agent.dsact import DSACTAgent, DSACTParams
from fpi_algorithm.algorithm.base import Algorithm
from fpi_algorithm.utils.experience import Experience


class DSACTAlgState(NamedTuple):
    q1_opt_state: optax.OptState
    q2_opt_state: optax.OptState
    pi_opt_state: optax.OptState
    log_alpha_opt_state: optax.OptState
    mean_q1_std: jax.Array
    mean_q2_std: jax.Array


class DSACT(Algorithm):
    def __init__(
        self,
        agent: DSACTAgent,
        *,
        gamma: float = 0.99,
        lr: float | optax.Schedule = 3e-4,
        max_grad_norm: Optional[float] = 40.,
        tau: float = 0.005,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        tau_b: float = 0.005,
        q_sample_clip: float = 3.0,
        bias: float = 0.1,
    ):
        self.agent = agent
        if target_entropy is None:
            target_entropy = -agent.act_dim

        optim = optax.adam(lr)
        if max_grad_norm is not None:
            optim = optax.chain(optax.clip_by_global_norm(max_grad_norm), optim)
        self.alg_state = DSACTAlgState(
            q1_opt_state=optim.init(agent.params.q1),
            q2_opt_state=optim.init(agent.params.q2),
            pi_opt_state=optim.init(agent.params.pi),
            log_alpha_opt_state=optim.init(agent.params.log_alpha),
            mean_q1_std=jnp.array(1, dtype=jnp.float32),
            mean_q2_std=jnp.array(1, dtype=jnp.float32),
        )

        @jax.jit
        def stateless_update(
            key: jax.Array,
            params: DSACTParams,
            alg_state: DSACTAlgState,
            data: Experience,
        ) -> Tuple[DSACTParams, DSACTAlgState, dict]:
            obs = data.obs
            action = data.action
            reward = data.reward
            next_obs = data.next_obs
            done = data.done

            q1_params = params.q1
            q2_params = params.q2
            target_q1_params = params.target_q1
            target_q2_params = params.target_q2
            pi_params = params.pi
            target_pi_params = params.target_pi
            log_alpha = params.log_alpha

            q1_opt_state = alg_state.q1_opt_state
            q2_opt_state = alg_state.q2_opt_state
            pi_opt_state = alg_state.pi_opt_state
            log_alpha_opt_state = alg_state.log_alpha_opt_state
            mean_q1_std = alg_state.mean_q1_std
            mean_q2_std = alg_state.mean_q2_std

            next_eval_key, q1_key, q2_key, eval_key = jax.random.split(key, 4)

            next_action, next_logp = agent.evaluate(next_eval_key, pi_params, next_obs)
            target_q1_mean, target_q1_std = agent.critic(target_q1_params, next_obs, next_action)
            target_q2_mean, target_q2_std = agent.critic(target_q2_params, next_obs, next_action)
            z1 = jnp.clip(jax.random.normal(q1_key, shape=target_q1_mean.shape), -q_sample_clip, q_sample_clip)
            z2 = jnp.clip(jax.random.normal(q2_key, shape=target_q2_mean.shape), -q_sample_clip, q_sample_clip)
            target_q1_sample = target_q1_mean + z1 * target_q1_std
            target_q2_sample = target_q2_mean + z2 * target_q2_std
            target_q = jnp.minimum(target_q1_mean, target_q2_mean)
            target_q_sample = jnp.where(target_q1_mean <= target_q2_mean, target_q1_sample, target_q2_sample)
            q_backup = reward + (1 - done) * gamma * (target_q - jnp.exp(log_alpha) * next_logp)
            q_backup_sample = reward + (1 - done) * gamma * (target_q_sample - jnp.exp(log_alpha) * next_logp)

            q1_mean, q1_std = agent.critic(q1_params, obs, action)
            q2_mean, q2_std = agent.critic(q2_params, obs, action)
            new_mean_q1_std = (1 - tau_b) * mean_q1_std + tau_b * q1_std.mean()
            new_mean_q2_std = (1 - tau_b) * mean_q2_std + tau_b * q2_std.mean()
            q1_diff = jnp.clip(q_backup_sample - q1_mean, -q_sample_clip * new_mean_q1_std, q_sample_clip * new_mean_q1_std)
            q2_diff = jnp.clip(q_backup_sample - q2_mean, -q_sample_clip * new_mean_q2_std, q_sample_clip * new_mean_q2_std)
            q1_backup_bound = q1_mean + q1_diff
            q2_backup_bound = q2_mean + q2_diff
            q1_mean_coef = (q_backup - q1_mean) / (q1_std ** 2 + bias)
            q2_mean_coef = (q_backup - q2_mean) / (q2_std ** 2 + bias)
            q1_std_coef = ((q1_mean - q1_backup_bound) ** 2 - q1_std ** 2) / (q1_std ** 3 + bias)
            q2_std_coef = ((q2_mean - q2_backup_bound) ** 2 - q2_std ** 2) / (q2_std ** 3 + bias)

            def q1_loss_fn(q1_params: hk.Params):
                q1_mean, q1_std = agent.critic(q1_params, obs, action)
                return (new_mean_q1_std ** 2 + bias) * (-q1_mean_coef * q1_mean - q1_std_coef * q1_std).mean()

            def q2_loss_fn(q2_params: hk.Params):
                q2_mean, q2_std = agent.critic(q2_params, obs, action)
                return (new_mean_q2_std ** 2 + bias) * (-q2_mean_coef * q2_mean - q2_std_coef * q2_std).mean()

            q1_loss, q1_grads = jax.value_and_grad(q1_loss_fn)(q1_params)
            q2_loss, q2_grads = jax.value_and_grad(q2_loss_fn)(q2_params)
            updates_q1, new_q1_opt_state = optim.update(q1_grads, q1_opt_state)
            updates_q2, new_q2_opt_state = optim.update(q2_grads, q2_opt_state)
            new_q1_params = optax.apply_updates(q1_params, updates_q1)
            new_q2_params = optax.apply_updates(q2_params, updates_q2)

            def pi_loss_fn(pi_params: hk.Params):
                action, logp = agent.evaluate(eval_key, pi_params, obs)
                q = jnp.minimum(
                    agent.critic(new_q1_params, obs, action)[0],
                    agent.critic(new_q2_params, obs, action)[0],
                )
                pi_loss = (jnp.exp(log_alpha) * logp - q).mean()
                return pi_loss, logp.mean()

            (pi_loss, logp), pi_grads = jax.value_and_grad(pi_loss_fn, has_aux=True)(pi_params)
            pi_update, new_pi_opt_state = optim.update(pi_grads, pi_opt_state)
            new_pi_params = optax.apply_updates(pi_params, pi_update)

            def log_alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
                return -(log_alpha * (logp + target_entropy)).mean()

            def log_alpha_update_fn(
                log_alpha: jax.Array,
                log_alpha_opt_state: optax.OptState,
            ) -> Tuple[jax.Array, optax.OptState]:
                log_alpha_grads = jax.grad(log_alpha_loss_fn)(log_alpha)
                log_alpha_update, log_alpha_opt_state = optim.update(
                    log_alpha_grads, log_alpha_opt_state)
                log_alpha = optax.apply_updates(log_alpha, log_alpha_update)
                return log_alpha, log_alpha_opt_state

            new_log_alpha, new_log_alpha_opt_state = jax.lax.cond(
                auto_alpha,
                log_alpha_update_fn,
                lambda *x: x,
                log_alpha,
                log_alpha_opt_state,
            )

            new_target_q1_params = optax.incremental_update(new_q1_params, target_q1_params, tau)
            new_target_q2_params = optax.incremental_update(new_q2_params, target_q2_params, tau)
            new_target_pi_params = optax.incremental_update(new_pi_params, target_pi_params, tau)

            params = DSACTParams(
                q1=new_q1_params,
                q2=new_q2_params,
                target_q1=new_target_q1_params,
                target_q2=new_target_q2_params,
                pi=new_pi_params,
                target_pi=new_target_pi_params,
                log_alpha=new_log_alpha,
            )
            alg_state = DSACTAlgState(
                q1_opt_state=new_q1_opt_state,
                q2_opt_state=new_q2_opt_state,
                pi_opt_state=new_pi_opt_state,
                log_alpha_opt_state=new_log_alpha_opt_state,
                mean_q1_std=new_mean_q1_std,
                mean_q2_std=new_mean_q2_std,
            )
            info = {
                'q1_loss': q1_loss,
                'q2_loss': q2_loss,
                'q1': q1_mean.mean(),
                'q2': q2_mean.mean(),
                'pi_loss': pi_loss,
                'entropy': -logp.mean(),
                'alpha': jnp.exp(log_alpha),
            }
            return params, alg_state, info

        self.stateless_update = stateless_update
