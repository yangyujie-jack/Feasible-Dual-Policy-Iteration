import os
from collections import deque
from typing import Callable, Dict, Optional, Tuple

import jax
import numpy as np
from tensorboardX import SummaryWriter
from gymnasium.vector import VectorEnv
from fpi_algorithm.algorithm.base import Algorithm
from fpi_algorithm.agent.base import Agent
from fpi_algorithm.buffer.buffer_is import BufferIS as Buffer
from fpi_algorithm.utils.random import seeding


class OffPolicyDualTrainer:
    def __init__(
        self,
        env: VectorEnv,
        agent: Agent,
        algorithm: Algorithm,
        buffer: Buffer,
        log_path: str,
        batch_size: int = 256,
        beta: float = 0.5,
        fea_window: int = 1000,
        dual_thresh: float = 0.95,
        start_step: int = 10000,
        total_step: int = int(1e6),
        sample_per_iteration: int = 1,
        update_per_iteration: int = 1,
        evaluate_env: Optional[VectorEnv] = None,
        evaluate_every: int = 10000,
        evaluate_n_episode: int = 10,
        sample_log_n_step: int = 10000,
        update_log_n_step: int = 1000,
        save_every: int = 100000,
        max_save_num: int = 1,
    ):
        self.env = env
        self.agent = agent
        self.algorithm = algorithm
        self.buffer = buffer
        self.log_path = log_path
        self.batch_size = batch_size
        self.beta = beta
        self.fea_window = fea_window
        self.dual_thresh = dual_thresh
        self.start_step = start_step
        self.total_step = total_step
        self.sample_per_iteration = sample_per_iteration
        self.update_per_iteration = update_per_iteration
        self.evaluate_env = evaluate_env
        self.evaluate_every = evaluate_every
        self.evaluate_n_episode = evaluate_n_episode
        self.sample_log_n_step = sample_log_n_step
        self.update_log_n_step = update_log_n_step
        self.save_every = save_every
        self.max_save_num = max_save_num
        self.save_steps = []

    def train(self, seed: int):
        key = jax.random.PRNGKey(seed)
        iter_key_fn = create_iter_key_fn(key)
        rng, _ = seeding(seed)

        sample_step = update_step = 0
        ep_ret = np.zeros(self.env.num_envs, dtype=np.float64)
        ep_cost = np.zeros(self.env.num_envs, dtype=np.float64)
        ep_len = np.zeros(self.env.num_envs, dtype=np.int64)
        cumulative_log_weight = np.zeros(self.env.num_envs, dtype=np.float32)
        cumulative_log_weight_dual = np.zeros(self.env.num_envs, dtype=np.float32)
        half_env_num = self.env.num_envs // 2
        fea = deque(maxlen=self.fea_window)

        sample_info = {
            'episode_return': [],
            'episode_cost': [],
            'episode_length': [],
            'IS_weight': [],
            'IS_weight_dual': [],
            'dual_activate': [],
        }
        update_info: Dict[str, list] = {}
        logger = SummaryWriter(self.log_path)

        action_space_seed = int(rng.integers(0, 2 ** 32 - 1))
        self.env.action_space.seed(action_space_seed)
        env_seed = int(rng.integers(0, 2 ** 32 - 1))
        obs, _ = self.env.reset(seed=env_seed)

        while sample_step < self.total_step:
            # setup random keys
            sample_key, update_key = iter_key_fn(sample_step)

            # sample data
            dual = len(fea) > 0 and np.mean(fea) > self.dual_thresh
            sample_info['dual_activate'].append(dual)
            for _ in range(self.sample_per_iteration):
                if sample_step < self.start_step:
                    action = self.env.action_space.sample()
                else:
                    act, dual_act, log_weight, log_weight_dual = self.agent.get_action(sample_key, obs, dual)
                    action = np.concatenate((act[:half_env_num], dual_act[half_env_num:]))

                next_obs, reward, cost, terminated, truncated, info = self.env.step(action)

                ep_ret += reward
                ep_cost += cost
                ep_len += 1
                done = terminated | truncated
                done_idx = np.where(done)[0]
                real_next_obs = next_obs.copy()
                if len(done_idx) > 0:
                    real_next_obs[done_idx] = np.stack(info['final_observation'][done_idx])
                    sample_info['episode_return'].extend(ep_ret[done_idx].tolist())
                    sample_info['episode_cost'].extend(ep_cost[done_idx].tolist())
                    sample_info['episode_length'].extend(ep_len[done_idx].tolist())
                    ep_ret[done_idx], ep_cost[done_idx], ep_len[done_idx] = 0., 0., 0

                self.buffer.add_batch(obs, action, real_next_obs, reward, cost, terminated,
                                      cumulative_log_weight, cumulative_log_weight_dual)

                if sample_step >= self.start_step:
                    cumulative_log_weight[:half_env_num] = \
                        self.beta * (cumulative_log_weight[:half_env_num] + log_weight[:half_env_num])
                    cumulative_log_weight_dual[half_env_num:] = \
                        self.beta * (cumulative_log_weight_dual[half_env_num:] + log_weight_dual[half_env_num:])
                    sample_info['IS_weight'].append(np.exp(log_weight[:half_env_num]).mean())
                    sample_info['IS_weight_dual'].append(np.exp(log_weight_dual[half_env_num:]).mean())
                    if len(done_idx) > 0:
                        cumulative_log_weight[done_idx] = 0.
                        cumulative_log_weight_dual[done_idx] = 0.

                for _ in range(self.env.num_envs):
                    sample_step += 1
                    if sample_step % self.sample_log_n_step == 0:
                        for k, v in sample_info.items():
                            if len(v) > 0:
                                logger.add_scalar(f'sample/{k}', np.mean(v), sample_step)
                                sample_info[k] = []
                        print('sample step', sample_step)

                obs = next_obs

            if sample_step < self.start_step:
                continue

            # update parameters
            for _ in range(self.update_per_iteration):
                data = self.buffer.sample(self.batch_size)
                alg_info = self.algorithm.update(update_key, data)
                fea.append(alg_info['feasible_ratio'])
                for k, v in alg_info.items():
                    if np.isnan(v):
                        continue
                    if k in update_info:
                        update_info[k].append(v)
                    else:
                        update_info[k] = [v]
                update_step += 1

                if update_step % self.update_log_n_step == 0:
                    for k, v in update_info.items():
                        logger.add_scalar(f'update/{k}', np.mean(v), update_step)
                        update_info[k] = []
                    print('update step', update_step)

                if update_step % self.save_every == 0:
                    self.save(update_step)

            # evaluate
            if self.evaluate_env is not None and sample_step % self.evaluate_every == 0:
                eval_info = self.evaluate(rng)
                for k, v in eval_info.items():
                    logger.add_scalar(f'evaluate/{k}', np.mean(v), sample_step)

        self.save(update_step)

    def evaluate(self, rng: np.random.Generator):
        eval_info = {
            'episode_return': [],
            'episode_cost': [],
            'episode_length': [],
        }
        ep_ret = np.zeros(self.evaluate_env.num_envs, dtype=np.float64)
        ep_cost = np.zeros(self.evaluate_env.num_envs, dtype=np.float64)
        ep_len = np.zeros(self.evaluate_env.num_envs, dtype=np.int64)
        ep_num = 0
        seed = int(rng.integers(0, 2 ** 32 - 1))
        obs, _ = self.evaluate_env.reset(seed=seed)
        while ep_num < self.evaluate_n_episode:
            action = self.agent.get_deterministic_action(obs)
            obs, reward, cost, terminated, truncated, _ = self.evaluate_env.step(action)
            ep_ret += reward
            ep_cost += cost
            ep_len += 1
            done_idx = np.where(terminated | truncated)[0]
            if len(done_idx) > 0:
                eval_info['episode_return'].extend(ep_ret[done_idx].tolist())
                eval_info['episode_cost'].extend(ep_cost[done_idx].tolist())
                eval_info['episode_length'].extend(ep_len[done_idx].tolist())
                ep_ret[done_idx], ep_cost[done_idx], ep_len[done_idx] = 0., 0., 0
                ep_num += len(done_idx)
        return eval_info

    def save(self, step: int):
        self.agent.save(os.path.join(self.log_path, f'params_{step}.pkl'))
        self.save_steps.append(step)
        if len(self.save_steps) > self.max_save_num:
            remove_step = self.save_steps.pop(0)
            os.remove(os.path.join(self.log_path, f'params_{remove_step}.pkl'))


def create_iter_key_fn(key: jax.Array) \
        -> Callable[[int], Tuple[jax.Array, jax.Array]]:
    def iter_key_fn(step: int):
        iter_key = jax.random.fold_in(key, step)
        sample_key, update_key = jax.random.split(iter_key)
        return sample_key, update_key

    iter_key_fn = jax.jit(iter_key_fn)
    iter_key_fn(0)  # Warm up
    return iter_key_fn
