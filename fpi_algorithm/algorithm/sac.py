from typing import NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from fpi_algorithm.agent.sac import SACAgent, SACParams
from fpi_algorithm.algorithm.base import Algorithm
from fpi_algorithm.utils.experience import Experience


class SACAlgState(NamedTuple):
    q1_opt_state: optax.OptState
    q2_opt_state: optax.OptState
    pi_opt_state: optax.OptState
    log_alpha_opt_state: optax.OptState


class SAC(Algorithm):
    def __init__(
        self,
        agent: SACAgent,
        *,
        gamma: float = 0.99,
        lr: float | optax.Schedule = 3e-4,
        max_grad_norm: Optional[float] = 40.,
        tau: float = 0.005,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
    ):
        self.agent = agent
        if target_entropy is None:
            target_entropy = -agent.act_dim

        optim = optax.adam(lr)
        if max_grad_norm is not None:
            optim = optax.chain(optax.clip_by_global_norm(max_grad_norm), optim)
        self.alg_state = SACAlgState(
            q1_opt_state=optim.init(agent.params.q1),
            q2_opt_state=optim.init(agent.params.q2),
            pi_opt_state=optim.init(agent.params.pi),
            log_alpha_opt_state=optim.init(agent.params.log_alpha),
        )

        @jax.jit
        def stateless_update(
            key: jax.Array,
            params: SACParams,
            alg_state: SACAlgState,
            data: Experience,
        ) -> Tuple[SACParams, SACAlgState, dict]:
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
            log_alpha = params.log_alpha

            q1_opt_state = alg_state.q1_opt_state
            q2_opt_state = alg_state.q2_opt_state
            pi_opt_state = alg_state.pi_opt_state
            log_alpha_opt_state = alg_state.log_alpha_opt_state

            next_eval_key, eval_key = jax.random.split(key)

            next_action, next_logp = agent.evaluate(next_eval_key, pi_params, next_obs)
            q1_target = agent.critic(target_q1_params, next_obs, next_action)
            q2_target = agent.critic(target_q2_params, next_obs, next_action)
            q_target = jnp.minimum(q1_target, q2_target) - jnp.exp(log_alpha) * next_logp
            q_backup = reward + (1 - done) * gamma * q_target

            def q_loss_fn(q_params: hk.Params):
                q = agent.critic(q_params, obs, action)
                return ((q - q_backup) ** 2).mean(), q

            (q1_loss, q1), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params)
            (q2_loss, q2), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params)
            q1_update, q1_opt_state = optim.update(q1_grads, q1_opt_state)
            q2_update, q2_opt_state = optim.update(q2_grads, q2_opt_state)
            new_q1_params = optax.apply_updates(q1_params, q1_update)
            new_q2_params = optax.apply_updates(q2_params, q2_update)

            def pi_loss_fn(pi_params: hk.Params):
                action, logp = agent.evaluate(eval_key, pi_params, obs)
                q = jnp.minimum(
                    agent.critic(new_q1_params, obs, action),
                    agent.critic(new_q2_params, obs, action),
                )
                pi_loss = (jnp.exp(log_alpha) * logp - q).mean()
                return pi_loss, logp.mean()

            (pi_loss, logp), pi_grads = jax.value_and_grad(pi_loss_fn, has_aux=True)(pi_params)
            pi_update, pi_opt_state = optim.update(pi_grads, pi_opt_state)
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

            params = SACParams(
                q1=new_q1_params,
                q2=new_q2_params,
                target_q1=new_target_q1_params,
                target_q2=new_target_q2_params,
                pi=new_pi_params,
                log_alpha=new_log_alpha,
            )
            alg_state = SACAlgState(
                q1_opt_state=q1_opt_state,
                q2_opt_state=q2_opt_state,
                pi_opt_state=pi_opt_state,
                log_alpha_opt_state=new_log_alpha_opt_state,
            )
            info = {
                'q1_loss': q1_loss,
                'q2_loss': q2_loss,
                'q1': q1.mean(),
                'q2': q2.mean(),
                'pi_loss': pi_loss,
                'entropy': -logp.mean(),
                'alpha': jnp.exp(log_alpha),
            }
            return params, alg_state, info

        self.stateless_update = stateless_update
