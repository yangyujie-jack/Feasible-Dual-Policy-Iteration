from typing import NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from fpi_algorithm.agent.sac_fpi import SACFPIAgent, SACFPIParams
from fpi_algorithm.algorithm.base import Algorithm
from fpi_algorithm.utils.experience import Experience


class SACFPIAlgState(NamedTuple):
    q1_opt_state: optax.OptState
    q2_opt_state: optax.OptState
    g1_opt_state: optax.OptState
    g2_opt_state: optax.OptState
    gr1_opt_state: optax.OptState
    gr2_opt_state: optax.OptState
    pi_opt_state: optax.OptState
    log_alpha_opt_state: optax.OptState
    log_cg_opt_state: optax.OptState
    lam1_opt_state: optax.OptState
    lam2_opt_state: optax.OptState


class SACFPI(Algorithm):
    def __init__(
        self,
        agent: SACFPIAgent,
        *,
        gamma: float = 0.99,
        cost_gamma: float = 0.97,
        lr: float | optax.Schedule = 3e-4,
        max_grad_norm: Optional[float] = 40.,
        tau: float = 0.005,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        pf: float = 0.1,
    ):
        self.agent = agent
        if target_entropy is None:
            target_entropy = -agent.act_dim

        optim = optax.adam(lr)
        if max_grad_norm is not None:
            optim = optax.chain(optax.clip_by_global_norm(max_grad_norm), optim)
        self.alg_state = SACFPIAlgState(
            q1_opt_state=optim.init(agent.params.q1),
            q2_opt_state=optim.init(agent.params.q2),
            g1_opt_state=optim.init(agent.params.g1),
            g2_opt_state=optim.init(agent.params.g2),
            gr1_opt_state=optim.init(agent.params.gr1),
            gr2_opt_state=optim.init(agent.params.gr2),
            pi_opt_state=optim.init(agent.params.pi),
            log_alpha_opt_state=optim.init(agent.params.log_alpha),
            log_cg_opt_state=optim.init(agent.params.log_cg),
            lam1_opt_state=optim.init(agent.params.lam1),
            lam2_opt_state=optim.init(agent.params.lam2),
        )

        @jax.jit
        def stateless_update(
            key: jax.Array,
            params: SACFPIParams,
            alg_state: SACFPIAlgState,
            data: Experience,
        ) -> Tuple[SACFPIParams, SACFPIAlgState, dict]:
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
            gr1_params = params.gr1
            gr2_params = params.gr2
            target_gr1_params = params.target_gr1
            target_gr2_params = params.target_gr2
            pi_params = params.pi
            log_alpha = params.log_alpha
            log_cg = params.log_cg
            lam1 = params.lam1
            lam2 = params.lam2

            q1_opt_state = alg_state.q1_opt_state
            q2_opt_state = alg_state.q2_opt_state
            g1_opt_state = alg_state.g1_opt_state
            g2_opt_state = alg_state.g2_opt_state
            gr1_opt_state = alg_state.gr1_opt_state
            gr2_opt_state = alg_state.gr2_opt_state
            pi_opt_state = alg_state.pi_opt_state
            log_alpha_opt_state = alg_state.log_alpha_opt_state
            log_cg_opt_state = alg_state.log_cg_opt_state
            lam1_opt_state = alg_state.lam1_opt_state
            lam2_opt_state = alg_state.lam2_opt_state

            next_eval_key, eval_key, new_eval_key = jax.random.split(key, 3)
            next_action, next_logp = agent.evaluate(next_eval_key, pi_params, next_obs)

            target_q1 = agent.critic(target_q1_params, next_obs, next_action)
            target_q2 = agent.critic(target_q2_params, next_obs, next_action)
            q_target = jnp.minimum(target_q1, target_q2) - jnp.exp(log_alpha) * next_logp
            q_backup = reward + (1 - done) * gamma * q_target

            def q_loss_fn(q_params: hk.Params):
                q = agent.critic(q_params, obs, action)
                return ((q - q_backup) ** 2).mean(), q

            (q1_loss, q1), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params)
            (q2_loss, q2), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params)
            update_q1, new_q1_opt_state = optim.update(q1_grads, q1_opt_state)
            update_q2, new_q2_opt_state = optim.update(q2_grads, q2_opt_state)
            new_q1_params = optax.apply_updates(q1_params, update_q1)
            new_q2_params = optax.apply_updates(q2_params, update_q2)

            target_g1 = agent.scenery(target_g1_params, next_obs, next_action)
            target_g2 = agent.scenery(target_g2_params, next_obs, next_action)
            target_g = jnp.clip(jnp.maximum(target_g1, target_g2), 0, 1)
            g_backup = cost + (1 - done) * (1 - cost) * cost_gamma * target_g

            def g_loss_fn(g_params: hk.Params):
                g = agent.scenery(g_params, obs, action)
                return ((g - g_backup) ** 2).mean(), g

            (g1_loss, g1), g1_grads = jax.value_and_grad(g_loss_fn, has_aux=True)(g1_params)
            (g2_loss, g2), g2_grads = jax.value_and_grad(g_loss_fn, has_aux=True)(g2_params)
            updates_g1, new_g1_opt_state = optim.update(g1_grads, g1_opt_state)
            updates_g2, new_g2_opt_state = optim.update(g2_grads, g2_opt_state)
            new_g1_params = optax.apply_updates(g1_params, updates_g1)
            new_g2_params = optax.apply_updates(g2_params, updates_g2)

            target_gr = jnp.clip(jnp.minimum(
                agent.scenery(target_gr1_params, next_obs, next_action),
                agent.scenery(target_gr2_params, next_obs, next_action),
            ), 0, 1)
            gr_backup = (1 - cost) + (1 - done) * cost * cost_gamma * target_gr

            def gr_loss_fn(gr_params: hk.Params):
                gr = agent.scenery(gr_params, obs, action)
                return ((gr - gr_backup) ** 2).mean(), gr

            (gr1_loss, gr1), gr1_grads = jax.value_and_grad(gr_loss_fn, has_aux=True)(gr1_params)
            (gr2_loss, gr2), gr2_grads = jax.value_and_grad(gr_loss_fn, has_aux=True)(gr2_params)
            updates_gr1, new_gr1_opt_state = optim.update(gr1_grads, gr1_opt_state)
            updates_gr2, new_gr2_opt_state = optim.update(gr2_grads, gr2_opt_state)
            new_gr1_params = optax.apply_updates(gr1_params, updates_gr1)
            new_gr2_params = optax.apply_updates(gr2_params, updates_gr2)

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
                gr = jnp.minimum(
                    agent.scenery(new_gr1_params, obs, action),
                    agent.scenery(new_gr2_params, obs, action),
                )
                vio = cost > 0
                fea = (g < pf) & ~vio
                cri = fea & (g >= pf - jnp.exp(log_cg))
                loss_fea = (fea & ~cri) * -q
                loss_cri = cri * (-q + lam1 * g) / (lam1 + 1)
                loss_inf = (~fea & ~vio) * (-q + lam2 * g) / (lam2 + 1)
                loss_vio = vio * -gr
                loss = (loss_fea + loss_cri + loss_inf + loss_vio + jnp.exp(log_alpha) * logp).mean()
                return loss, (fea, cri, g, logp.mean())

            (pi_loss, (fea, cri, g, logp)), pi_grads = jax.value_and_grad(pi_loss_fn, has_aux=True)(pi_params)
            updates_pi, new_pi_opt_state = optim.update(pi_grads, pi_opt_state)
            new_pi_params = optax.apply_updates(pi_params, updates_pi)

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

            new_action, _ = agent.evaluate(new_eval_key, new_pi_params, obs)
            new_g = jnp.maximum(
                agent.scenery(new_g1_params, obs, new_action),
                agent.scenery(new_g2_params, obs, new_action),
            )
            fea_ratio = fea.mean()
            vio = (fea & (new_g > pf))
            vio_ratio = vio.mean()
            delta_cg = masked_mean(jax.nn.leaky_relu((pf - g) - jnp.exp(log_cg)), vio) + \
                ((fea_ratio > 0) & (vio_ratio == 0)) * jax.nn.leaky_relu(-jnp.exp(log_cg))
            fea_g_vio = masked_mean(jax.nn.leaky_relu(new_g - pf), cri)
            inf_g_inc = masked_mean(jax.nn.leaky_relu(new_g - g), ~fea)

            log_cg_grads = -delta_cg
            updates_log_cg, new_log_cg_opt_state = optim.update(log_cg_grads, log_cg_opt_state)
            new_log_cg = optax.apply_updates(log_cg, updates_log_cg)

            lam1_grads = -fea_g_vio
            updates_lam1, new_lam1_opt_state = optim.update(lam1_grads, lam1_opt_state)
            new_lam1 = optax.apply_updates(lam1, updates_lam1)
            new_lam1 = jnp.maximum(new_lam1, 0)

            lam2_grads = -inf_g_inc
            updates_lam2, new_lam2_opt_state = optim.update(lam2_grads, lam2_opt_state)
            new_lam2 = optax.apply_updates(lam2, updates_lam2)
            new_lam2 = jnp.maximum(new_lam2, 0)

            new_target_q1_params = optax.incremental_update(new_q1_params, target_q1_params, tau)
            new_target_q2_params = optax.incremental_update(new_q2_params, target_q2_params, tau)
            new_target_g1_params = optax.incremental_update(new_g1_params, target_g1_params, tau)
            new_target_g2_params = optax.incremental_update(new_g2_params, target_g2_params, tau)
            new_target_gr1_params = optax.incremental_update(new_gr1_params, target_gr1_params, tau)
            new_target_gr2_params = optax.incremental_update(new_gr2_params, target_gr2_params, tau)

            new_params = SACFPIParams(
                q1=new_q1_params,
                q2=new_q2_params,
                target_q1=new_target_q1_params,
                target_q2=new_target_q2_params,
                g1=new_g1_params,
                g2=new_g2_params,
                target_g1=new_target_g1_params,
                target_g2=new_target_g2_params,
                gr1=new_gr1_params,
                gr2=new_gr2_params,
                target_gr1=new_target_gr1_params,
                target_gr2=new_target_gr2_params,
                pi=new_pi_params,
                log_alpha=new_log_alpha,
                log_cg=new_log_cg,
                lam1=new_lam1,
                lam2=new_lam2,
            )
            new_alg_state = SACFPIAlgState(
                q1_opt_state=new_q1_opt_state,
                q2_opt_state=new_q2_opt_state,
                g1_opt_state=new_g1_opt_state,
                g2_opt_state=new_g2_opt_state,
                gr1_opt_state=new_gr1_opt_state,
                gr2_opt_state=new_gr2_opt_state,
                pi_opt_state=new_pi_opt_state,
                log_alpha_opt_state=new_log_alpha_opt_state,
                log_cg_opt_state=new_log_cg_opt_state,
                lam1_opt_state=new_lam1_opt_state,
                lam2_opt_state=new_lam2_opt_state,
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
                'gr1_loss': gr1_loss,
                'gr2_loss': gr2_loss,
                'gr1': gr1.mean(),
                'gr2': gr2.mean(),
                'pi_loss': pi_loss,
                'entropy': -logp.mean(),
                'alpha': jnp.exp(log_alpha),
                'feasible_ratio': fea_ratio,
                'critical_ratio': cri.mean(),
                'cg': jnp.exp(log_cg),
                'lam1': lam1,
                'lam2': lam2,
                'feasible_g_violation_ratio': vio_ratio,
                'feasible_g_violation': fea_g_vio,
                'infeasible_g_increment': inf_g_inc,
                'violate_ratio': cost.mean(),
            }
            return new_params, new_alg_state, info

        self.stateless_update = stateless_update


def masked_mean(x: jax.Array, mask: jax.Array) -> jax.Array:
    return (x * mask).sum() / jnp.maximum(mask.sum(), 1)
