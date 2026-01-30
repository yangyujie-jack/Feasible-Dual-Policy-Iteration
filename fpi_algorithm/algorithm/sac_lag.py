from typing import NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from fpi_algorithm.agent.sac_lag import SACLagAgent, SACLagParams
from fpi_algorithm.algorithm.base import Algorithm
from fpi_algorithm.utils.experience import Experience


class SACLagAlgState(NamedTuple):
    q1_opt_state: optax.OptState
    q2_opt_state: optax.OptState
    g1_opt_state: optax.OptState
    g2_opt_state: optax.OptState
    pi_opt_state: optax.OptState
    log_alpha_opt_state: optax.OptState
    lam_opt_state: optax.OptState
    step: int


class SACLag(Algorithm):
    def __init__(
        self,
        agent: SACLagAgent,
        *,
        gamma: float = 0.99,
        lr: float | optax.Schedule = 3e-4,
        max_grad_norm: Optional[float] = 40.,
        tau: float = 0.005,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        cost_lim: float = 0.1,
        multiplier_delay: int = 10,
    ):
        self.agent = agent
        if target_entropy is None:
            target_entropy = -agent.act_dim

        optim = optax.adam(lr)
        if max_grad_norm is not None:
            optim = optax.chain(optax.clip_by_global_norm(max_grad_norm), optim)
        self.alg_state = SACLagAlgState(
            q1_opt_state=optim.init(agent.params.q1),
            q2_opt_state=optim.init(agent.params.q2),
            g1_opt_state=optim.init(agent.params.g1),
            g2_opt_state=optim.init(agent.params.g2),
            pi_opt_state=optim.init(agent.params.pi),
            log_alpha_opt_state=optim.init(agent.params.log_alpha),
            lam_opt_state=optim.init(agent.params.lam),
            step=0,
        )

        @jax.jit
        def stateless_update(
            key: jax.Array,
            params: SACLagParams,
            alg_state: SACLagAlgState,
            data: Experience,
        ) -> Tuple[SACLagParams, SACLagAlgState, dict]:
            obs = data.obs
            action = data.action
            reward = data.reward
            cost = data.cost
            next_obs = data.next_obs
            done = data.done

            q1_params = params.q1
            q2_params = params.q2
            target_q1_params = params.target_q1
            target_q2_params = params.target_q2
            g1_params = params.g1
            g2_params = params.g2
            target_g1_params = params.target_g1
            target_g2_params = params.target_g2
            pi_params = params.pi
            log_alpha = params.log_alpha
            lam = params.lam

            q1_opt_state = alg_state.q1_opt_state
            q2_opt_state = alg_state.q2_opt_state
            g1_opt_state = alg_state.g1_opt_state
            g2_opt_state = alg_state.g2_opt_state
            pi_opt_state = alg_state.pi_opt_state
            log_alpha_opt_state = alg_state.log_alpha_opt_state
            lam_opt_state = alg_state.lam_opt_state
            step = alg_state.step

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

            target_g1 = agent.scenery(target_g1_params, next_obs, next_action)
            target_g2 = agent.scenery(target_g2_params, next_obs, next_action)
            target_g = jnp.maximum(target_g1, target_g2)
            g_backup = cost + (1 - done) * gamma * target_g

            def g_loss_fn(g_params: hk.Params):
                g = agent.scenery(g_params, obs, action)
                return ((g - g_backup) ** 2).mean(), g

            (g1_loss, g1), g1_grads = jax.value_and_grad(g_loss_fn, has_aux=True)(g1_params)
            (g2_loss, g2), g2_grads = jax.value_and_grad(g_loss_fn, has_aux=True)(g2_params)
            updates_g1, new_g1_opt_state = optim.update(g1_grads, g1_opt_state)
            updates_g2, new_g2_opt_state = optim.update(g2_grads, g2_opt_state)
            new_g1_params = optax.apply_updates(g1_params, updates_g1)
            new_g2_params = optax.apply_updates(g2_params, updates_g2)

            def pi_loss_fn(pi_params: hk.Params):
                action, logp = agent.evaluate(eval_key, pi_params, obs)
                q = jnp.minimum(
                    agent.critic(new_q1_params, obs, action),
                    agent.critic(new_q2_params, obs, action),
                )
                g = jnp.maximum(
                    agent.scenery(new_g1_params, obs, action),
                    agent.scenery(new_g2_params, obs, action),
                )
                pi_loss = ((jnp.exp(log_alpha) * logp - q + lam * g) / (lam + 1)).mean()
                return pi_loss, (g.mean(), logp.mean())

            (pi_loss, (g, logp)), pi_grads = jax.value_and_grad(pi_loss_fn, has_aux=True)(pi_params)
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

            def lam_update_fn():
                lam_grads = cost_lim - g
                updates_lam, new_lam_opt_state = optim.update(lam_grads, lam_opt_state)
                new_lam = optax.apply_updates(lam, updates_lam)
                return new_lam, new_lam_opt_state

            new_lam, new_lam_opt_state = jax.lax.cond(
                step % multiplier_delay == 0,
                lam_update_fn,
                lambda: (lam, lam_opt_state),
            )

            new_target_q1_params = optax.incremental_update(new_q1_params, target_q1_params, tau)
            new_target_q2_params = optax.incremental_update(new_q2_params, target_q2_params, tau)
            new_target_g1_params = optax.incremental_update(new_g1_params, target_g1_params, tau)
            new_target_g2_params = optax.incremental_update(new_g2_params, target_g2_params, tau)

            params = SACLagParams(
                q1=new_q1_params,
                q2=new_q2_params,
                target_q1=new_target_q1_params,
                target_q2=new_target_q2_params,
                g1=new_g1_params,
                g2=new_g2_params,
                target_g1=new_target_g1_params,
                target_g2=new_target_g2_params,
                pi=new_pi_params,
                log_alpha=new_log_alpha,
                lam=new_lam,
            )
            alg_state = SACLagAlgState(
                q1_opt_state=q1_opt_state,
                q2_opt_state=q2_opt_state,
                g1_opt_state=new_g1_opt_state,
                g2_opt_state=new_g2_opt_state,
                pi_opt_state=pi_opt_state,
                log_alpha_opt_state=new_log_alpha_opt_state,
                lam_opt_state=new_lam_opt_state,
                step=step + 1,
            )
            info = {
                'q1_loss': q1_loss,
                'q2_loss': q2_loss,
                'q1': q1.mean(),
                'q2': q2.mean(),
                'g1_loss': g1_loss,
                'g2_loss': g2_loss,
                'g1': g1.mean(),
                'g2': g2.mean(),
                'pi_loss': pi_loss,
                'entropy': -logp.mean(),
                'alpha': jnp.exp(log_alpha),
                'lam': lam,
                'violate_ratio': cost.mean(),
            }
            return params, alg_state, info

        self.stateless_update = stateless_update
