import argparse
import json
import os
import time

import jax
from fpi_algorithm.agent.td3 import TD3Agent
from fpi_algorithm.algorithm.td3 import TD3
from fpi_algorithm.agent.sac import SACAgent
from fpi_algorithm.algorithm.sac import SAC
from fpi_algorithm.algorithm.sac_pen import SACPen
from fpi_algorithm.agent.dsact import DSACTAgent
from fpi_algorithm.algorithm.dsact import DSACT
from fpi_algorithm.algorithm.dsact_pen import DSACTPen
from fpi_algorithm.agent.sac_lag import SACLagAgent
from fpi_algorithm.algorithm.sac_lag import SACLag
from fpi_algorithm.agent.td3_lag import TD3LagAgent
from fpi_algorithm.algorithm.td3_lag import TD3Lag
from fpi_algorithm.agent.sac_fpi import SACFPIAgent
from fpi_algorithm.algorithm.sac_fpi import SACFPI
from fpi_algorithm.agent.sac_fpi_dual import SACFPIDualAgent
from fpi_algorithm.algorithm.sac_fpi_dual import SACFPIDual
from fpi_algorithm.buffer.buffer import Buffer
from fpi_algorithm.buffer.buffer_is import BufferIS
from fpi_algorithm.trainer.off_policy import OffPolicyTrainer
from fpi_algorithm.trainer.off_policy_dual import OffPolicyDualTrainer
from fpi_algorithm.utils.path import PROJECT_ROOT
from fpi_algorithm.utils.random import seeding
from fpi_algorithm.utils.env import make_env


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # common
    parser.add_argument('--env', type=str, default='SafetyPointGoal1-v0')
    parser.add_argument('--alg', type=str, default='SACFPIDual')
    parser.add_argument('--env_num', type=int, default=2)
    parser.add_argument('--eval_env_num', type=int, default=2)
    parser.add_argument('--hidden_num', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--target_entropy', type=float, default=None)
    parser.add_argument('--total_step', type=int, default=int(2e6))
    parser.add_argument('--buffer_size', type=int, default=int(2e6))
    parser.add_argument('--evaluate_every', type=int, default=int(2e4))
    parser.add_argument('--save_every', type=int, default=int(2e5))
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--note', type=str, default='')
    parser.add_argument('--GPU_memory', type=str, default='.1')

    # penalty
    parser.add_argument('--penalty_coef', type=float, default=1.)

    # Lagrangian
    parser.add_argument('--multiplier_delay', type=int, default=10)

    # FDPI
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--dual_thresh', type=float, default=0.95)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--target_kl', type=float, default=5.0)

    args = parser.parse_args()

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = args.GPU_memory

    master_seed = args.seed
    master_rng, _ = seeding(master_seed)
    init_network_seed, buffer_seed, train_seed = map(
        int, master_rng.integers(0, 2 ** 32 - 1, 3))

    env = make_env(args.env, env_num=args.env_num, train=True)
    eval_env = make_env(args.env, env_num=args.eval_env_num)

    buffer_args = dict(
        obs_dim=env.observation_space.shape[-1],
        act_dim=env.action_space.shape[-1],
        size=args.buffer_size,
        seed=buffer_seed,
    )
    if 'Dual' in args.alg:
        buffer_cls = BufferIS
    else:
        buffer_cls = Buffer
    buffer = buffer_cls(**buffer_args)

    init_network_key = jax.random.PRNGKey(init_network_seed)

    agent_args = dict(
        key=init_network_key,
        obs_dim=env.observation_space.shape[-1],
        act_dim=env.action_space.shape[-1],
        hidden_sizes=[args.hidden_dim] * args.hidden_num,
    )
    if args.alg == 'TD3':
        agent_cls = TD3Agent
    elif args.alg == 'SAC':
        agent_cls = SACAgent
    elif args.alg == 'DSACT':
        agent_cls = DSACTAgent
    elif args.alg == 'SACPen':
        agent_cls = SACAgent
    elif args.alg == 'DSACTPen':
        agent_cls = DSACTAgent
    elif args.alg == 'TD3Lag':
        agent_cls = TD3LagAgent
    elif args.alg == 'SACLag':
        agent_cls = SACLagAgent
    elif args.alg == 'SACFPI':
        agent_cls = SACFPIAgent
    elif args.alg == 'SACFPIDual':
        agent_cls = SACFPIDualAgent
    else:
        raise ValueError(f'Invalid algorithm {args.alg}!')
    agent = agent_cls(**agent_args)

    alg_args = dict(agent=agent, lr=args.lr)
    if 'SAC' in args.alg:
        alg_args.update(target_entropy=args.target_entropy)
    if 'Pen' in args.alg:
        alg_args.update(penalty_coef=args.penalty_coef)
    if 'Lag' in args.alg:
        alg_args.update(multiplier_delay=args.multiplier_delay)
    if args.alg == 'TD3':
        alg_cls = TD3
    elif args.alg == 'SAC':
        alg_cls = SAC
    elif args.alg == 'DSACT':
        alg_cls = DSACT
    elif args.alg == 'SACPen':
        alg_cls = SACPen
    elif args.alg == 'DSACTPen':
        alg_cls = DSACTPen
    elif args.alg == 'TD3Lag':
        alg_cls = TD3Lag
    elif args.alg == 'SACLag':
        alg_cls = SACLag
    elif args.alg == 'SACFPI':
        alg_cls = SACFPI
    elif args.alg == 'SACFPIDual':
        alg_cls = SACFPIDual
        alg_args.update(pf=args.epsilon, target_kl=args.target_kl)
    algorithm = alg_cls(**alg_args)

    logdir_name = args.alg
    if len(args.note) > 0:
        logdir_name += '_' + args.note
    if args.epsilon != 0.1:
        logdir_name += f'_epsilon{args.epsilon}'
    if args.target_kl != 5.0:
        logdir_name += f'_target_kl{args.target_kl}'
    logdir_name += f'_seed{args.seed}_' + time.strftime('%Y-%m-%d_%H-%M-%S')
    log_path = os.path.join(PROJECT_ROOT, 'log', args.env, logdir_name)
    os.makedirs(log_path, exist_ok=True)

    with open(os.path.join(log_path, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    trainer_args = dict(
        env=env,
        agent=agent,
        algorithm=algorithm,
        buffer=buffer,
        log_path=log_path,
        total_step=args.total_step,
        evaluate_env=eval_env,
        evaluate_every=args.evaluate_every,
        save_every=args.save_every,
    )
    if 'Dual' in args.alg:
        trainer_cls = OffPolicyDualTrainer
        trainer_args.update(
            beta=args.beta,
            dual_thresh=args.dual_thresh,
        )
    else:
        trainer_cls = OffPolicyTrainer
    trainer = trainer_cls(**trainer_args)

    trainer.train(train_seed)
