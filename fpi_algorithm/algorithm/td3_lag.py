from typing import NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from fpi_algorithm.agent.td3_lag import TD3LagAgent, TD3LagParams 
from fpi_algorithm.algorithm.base import Algorithm
from fpi_algorithm.utils.experience import Experience


class TD3LagAlgState(NamedTuple):
    q1_opt_state: optax.OptState
    q2_opt_state: optax.OptState
    g1_opt_state: optax.OptState
    g2_opt_state: optax.OptState
    pi_opt_state: optax.OptState
    lam_opt_state: optax.OptState
    step: int


class TD3Lag(Algorithm):
    def __init__(
        self,
        agent: TD3LagAgent,
        *,
        gamma: float = 0.99,
        lr: float | optax.Schedule = 3e-4,
        max_grad_norm: Optional[float] = 40.,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_update_freq: int = 2,
        cost_lim: float = 0.1,
        multiplier_delay: int = 10,
    ):
        self.agent = agent

        optim = optax.adam(lr)
        if max_grad_norm is not None:
            optim = optax.chain(optax.clip_by_global_norm(max_grad_norm), optim)
        self.alg_state = TD3LagAlgState(
            q1_opt_state=optim.init(agent.params.q1),
            q2_opt_state=optim.init(agent.params.q2),
            pi_opt_state=optim.init(agent.params.pi),
            g1_opt_state=optim.init(agent.params.g1),
            g2_opt_state=optim.init(agent.params.g2),
            lam_opt_state=optim.init(agent.params.lam),
            step=0,
        )

        @jax.jit
        def stateless_update(
            key: jax.Array,
            params: TD3LagParams,
            alg_state: TD3LagAlgState,
            data: Experience,
        ) -> Tuple[TD3LagParams, TD3LagAlgState, dict]:
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
            target_pi_params = params.target_pi
            lam = params.lam

            q1_opt_state = alg_state.q1_opt_state
            q2_opt_state = alg_state.q2_opt_state
            g1_opt_state = alg_state.g1_opt_state
            g2_opt_state = alg_state.g2_opt_state
            pi_opt_state = alg_state.pi_opt_state
            lam_opt_state = alg_state.lam_opt_state
            step = alg_state.step

            noise = jax.random.normal(key, shape=action.shape) * policy_noise
            noise = jnp.clip(noise, -noise_clip, noise_clip)
            next_action = agent.actor(target_pi_params, next_obs) + noise
            next_action = jnp.clip(next_action, -1.0, 1.0)

            target_q1 = agent.critic(target_q1_params, next_obs, next_action)
            target_q2 = agent.critic(target_q2_params, next_obs, next_action)
            target_q = jnp.minimum(target_q1, target_q2)
            q_backup = reward + (1 - done) * gamma * target_q

            def q_loss_fn(p: hk.Params):
                q = agent.critic(p, obs, action)
                loss_q = ((q - q_backup) ** 2).mean()
                return loss_q, q

            (q1_loss, q1), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params)
            (q2_loss, q2), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params)
            updates_q1, new_q1_opt_state = optim.update(q1_grads, q1_opt_state)
            updates_q2, new_q2_opt_state = optim.update(q2_grads, q2_opt_state)
            new_q1_params = optax.apply_updates(q1_params, updates_q1)
            new_q2_params = optax.apply_updates(q2_params, updates_q2)

            target_g1 = agent.scenery(target_g1_params, next_obs, next_action)
            target_g2 = agent.scenery(target_g2_params, next_obs, next_action)
            target_g = jnp.maximum(target_g1, target_g2)
            g_backup = cost + (1.0 - done) * gamma * target_g

            def g_loss_fn(p: hk.Params):
                g = agent.scenery(p, obs, action)
                loss_g = ((g - g_backup) ** 2).mean()
                return loss_g, g

            (g1_loss, g1), g1_grads = jax.value_and_grad(g_loss_fn, has_aux=True)(g1_params)
            (g2_loss, g2), g2_grads = jax.value_and_grad(g_loss_fn, has_aux=True)(g2_params)
            updates_g1, new_g1_opt_state = optim.update(g1_grads, g1_opt_state)
            updates_g2, new_g2_opt_state = optim.update(g2_grads, g2_opt_state)
            new_g1_params = optax.apply_updates(g1_params, updates_g1)
            new_g2_params = optax.apply_updates(g2_params, updates_g2)

            def update_pi_fn(pi_params: hk.Params, pi_opt_state: optax.OptState):
                def pi_loss_fn(pi_params: hk.Params):
                    act = agent.actor(pi_params, obs)
                    q = jnp.minimum(
                        agent.critic(new_q1_params, obs, act),
                        agent.critic(new_q2_params, obs, act),
                    )
                    g = jnp.maximum(
                        agent.scenery(new_g1_params, obs, act),
                        agent.scenery(new_g2_params, obs, act),
                    )
                    pi_loss = ((-q + lam * g) / (lam + 1)).mean()
                    return pi_loss, g.mean()
                (pi_loss, g), pi_grads = jax.value_and_grad(pi_loss_fn, has_aux=True)(pi_params)
                updates_actor, new_pi_opt_state = optim.update(pi_grads, pi_opt_state)
                new_pi_params = optax.apply_updates(pi_params, updates_actor)
                return new_pi_params, new_pi_opt_state, pi_loss, g

            do_update_actor = (step % policy_update_freq) == 0

            new_pi_params, new_pi_opt_state, pi_loss, g = jax.lax.cond(
                do_update_actor,
                update_pi_fn,
                lambda *x: (*x, jnp.nan, jnp.nan),
                pi_params, pi_opt_state,
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

            new_target_q1_params = jax.lax.cond(
                do_update_actor,
                lambda: optax.incremental_update(new_q1_params, target_q1_params, tau),
                lambda: target_q1_params
            )
            new_target_q2_params = jax.lax.cond(
                do_update_actor,
                lambda: optax.incremental_update(new_q2_params, target_q2_params, tau),
                lambda: target_q2_params
            )
            new_target_g1_params = jax.lax.cond(
                do_update_actor,
                lambda: optax.incremental_update(new_g1_params, target_g1_params, tau),
                lambda: target_g1_params
            )
            new_target_g2_params = jax.lax.cond(
                do_update_actor,
                lambda: optax.incremental_update(new_g2_params, target_g2_params, tau),
                lambda: target_g2_params
            )
            new_target_pi_params = jax.lax.cond(
                do_update_actor,
                lambda: optax.incremental_update(new_pi_params, target_pi_params, tau),
                lambda: target_pi_params
            )

            new_params = TD3LagParams(
                q1=new_q1_params,
                q2=new_q2_params,
                target_q1=new_target_q1_params,
                target_q2=new_target_q2_params,
                g1=new_g1_params,
                g2=new_g2_params,
                target_g1=new_target_g1_params,
                target_g2=new_target_g2_params,
                pi=new_pi_params,
                target_pi=new_target_pi_params,
                lam=new_lam,
            )
            new_alg_state = TD3LagAlgState(
                q1_opt_state=new_q1_opt_state,
                q2_opt_state=new_q2_opt_state,
                pi_opt_state=new_pi_opt_state,
                g1_opt_state=new_g1_opt_state,
                g2_opt_state=new_g2_opt_state,
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
                'lam': lam,
                'violate_ratio': cost.mean(),
            }
            return new_params, new_alg_state, info

        self.stateless_update = stateless_update
