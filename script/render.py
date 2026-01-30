import argparse
import json
import os
from datetime import datetime

import jax
import gymnasium as gym
from fpi_algorithm.agent.sac_fpi import SACFPIAgent
from fpi_algorithm.utils.path import PROJECT_ROOT
from fpi_algorithm.utils.wrapper import NormalizeStepOutputWrapper
from fpi_algorithm.utils.env import make_env


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='')
    parser.add_argument('--step', type=int, default=995001)
    parser.add_argument('--seed', type=int, default=1)  
    parser.add_argument('--GPU_memory', type=str, default='.1')
    args = parser.parse_args()

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = args.GPU_memory

    log_path = os.path.join(PROJECT_ROOT, args.dir)
    video_folder = os.path.join(log_path, 'videos', datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

    with open(os.path.join(log_path, 'config.json'), 'r') as f:
        config = json.load(f)

    env = make_env(config['env'], render_mode='rgb_array', camera_id=1)
    env = gym.wrappers.RecordVideo(
        NormalizeStepOutputWrapper(env),
        video_folder=video_folder,
        name_prefix=f'seed{args.seed}',
        episode_trigger=lambda _: True,
    )

    init_network_key = jax.random.PRNGKey(0)
    agent = SACFPIAgent(
        key=init_network_key,
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        hidden_sizes=[config['hidden_dim']] * config['hidden_num'],
    )
    agent.load(os.path.join(log_path, f'params_{args.step}.pkl'))

    obs, _ = env.reset(seed=args.seed)
    while True:
        action = agent.get_deterministic_action(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()
