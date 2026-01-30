from typing import NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from fpi_algorithm.agent.td3 import TD3Agent, TD3Params 
from fpi_algorithm.algorithm.base import Algorithm
from fpi_algorithm.utils.experience import Experience


class TD3AlgState(NamedTuple):
    q1_opt_state: optax.OptState
    q2_opt_state: optax.OptState
    pi_opt_state: optax.OptState
    step: int


class TD3(Algorithm):
    def __init__(
        self,
        agent: TD3Agent,
        *,
        gamma: float = 0.99,
        lr: float | optax.Schedule = 3e-4,
        max_grad_norm: Optional[float] = 40.,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_update_freq: int = 2,
    ):
        self.agent = agent
        policy_update_freq = policy_update_freq

        optim = optax.adam(lr)
        if max_grad_norm is not None:
            optim = optax.chain(optax.clip_by_global_norm(max_grad_norm), optim)
        self.alg_state = TD3AlgState(
            q1_opt_state=optim.init(agent.params.q1),
            q2_opt_state=optim.init(agent.params.q2),
            pi_opt_state=optim.init(agent.params.pi),
            step=0,
        )

        @jax.jit
        def stateless_update(
            key: jax.Array,
            params: TD3Params,
            alg_state: TD3AlgState,
            data: Experience
        ) -> Tuple[TD3Params, TD3AlgState, dict]:
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

            q1_opt_state = alg_state.q1_opt_state
            q2_opt_state = alg_state.q2_opt_state
            pi_opt_state = alg_state.pi_opt_state
            step = alg_state.step

            noise = jax.random.normal(key, shape=action.shape) * policy_noise
            noise = jnp.clip(noise, -noise_clip, noise_clip)
            next_action = agent.actor(target_pi_params, next_obs) + noise
            next_action = jnp.clip(next_action, -1.0, 1.0)

            target_q1 = agent.critic(target_q1_params, next_obs, next_action)
            target_q2 = agent.critic(target_q2_params, next_obs, next_action)
            target_q = jnp.minimum(target_q1, target_q2)
            q_backup = reward + (1 - done) * gamma * target_q

            def q_loss_fn(q_params: hk.Params):
                q = agent.critic(q_params, obs, action)
                loss = ((q - q_backup) ** 2).mean()
                return loss, q

            (q1_loss, q1), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params)
            (q2_loss, q2), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params)
            updates_q1, new_q1_opt_state = optim.update(q1_grads, q1_opt_state)
            updates_q2, new_q2_opt_state = optim.update(q2_grads, q2_opt_state)
            new_q1_params = optax.apply_updates(q1_params, updates_q1)
            new_q2_params = optax.apply_updates(q2_params, updates_q2)

            def update_pi_fn(pi_params: hk.Params, pi_opt_state: optax.OptState):
                def pi_loss_fn(pi_params: hk.Params):
                    act = agent.actor(pi_params, obs)
                    return -jnp.minimum(
                        agent.critic(new_q1_params, obs, act),
                        agent.critic(new_q2_params, obs, act),
                    ).mean()
                pi_loss, actor_grads = jax.value_and_grad(pi_loss_fn)(pi_params)
                updates_actor, new_pi_opt_state = optim.update(actor_grads, pi_opt_state)
                new_pi_params = optax.apply_updates(pi_params, updates_actor)
                return new_pi_params, new_pi_opt_state, pi_loss

            do_update_actor = (step % policy_update_freq) == 0

            new_pi_params, new_pi_opt_state, pi_loss = jax.lax.cond(
                do_update_actor,
                update_pi_fn,
                lambda *x: (*x, jnp.nan),
                pi_params, pi_opt_state,
            )

            new_target_q1_params = jax.lax.cond(
                do_update_actor,
                lambda: optax.incremental_update(new_q1_params, target_q1_params, tau),
                lambda: target_q1_params,
            )
            new_target_q2_params = jax.lax.cond(
                do_update_actor,
                lambda: optax.incremental_update(new_q2_params, target_q2_params, tau),
                lambda: target_q2_params,
            )
            new_target_pi_params = jax.lax.cond(
                do_update_actor,
                lambda: optax.incremental_update(new_pi_params, target_pi_params, tau),
                lambda: target_pi_params,
            )

            new_params = TD3Params(
                q1=new_q1_params,
                q2=new_q2_params,
                target_q1=new_target_q1_params,
                target_q2=new_target_q2_params,
                pi=new_pi_params,
                target_pi=new_target_pi_params,
            )
            new_alg_state = TD3AlgState(
                q1_opt_state=new_q1_opt_state,
                q2_opt_state=new_q2_opt_state,
                pi_opt_state=new_pi_opt_state,
                step=step + 1,
            )

            info = {
                'q1_loss': q1_loss,
                'q2_loss': q2_loss,
                'q1': q1.mean(),
                'q2': q2.mean(),
                'pi_loss': pi_loss,
            }
            return new_params, new_alg_state, info

        self.stateless_update = stateless_update
