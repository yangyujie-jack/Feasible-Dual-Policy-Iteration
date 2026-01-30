import json
import os
from typing import List, Sequence, Union

import jax
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.ticker import MaxNLocator, LogLocator
from tensorboard.backend.event_processing import event_accumulator
from fpi_algorithm.agent.sac_fpi import SACFPIAgent
from fpi_algorithm.agent.sac_fpi_dual import SACFPIDualAgent
from fpi_algorithm.utils.env import make_env
from fpi_algorithm.utils.path import PROJECT_ROOT, LOG_PATH, RESULT_PATH, FIGURE_PATH


EXTENSION = 'png'

TBTAGS = {
    'cost': 'evaluate/episode_cost',
    'return': 'evaluate/episode_return',
    'violation': 'update/violate_ratio',
    'dual': 'sample/dual_activate',
}

TAGLABELS = {
    'cost': 'Episode cost',
    'return': 'Episode return',
}

ALGLABELCOLORS = {
    'RCPO': ('RCPO', 'C0'),
    'FOCOPS': ('FOCOPS', 'C1'),
    'CUP': ('CUP', 'C2'),
    'SACLag': ('SAC-Lag', 'C4'),
    'DSACTPen': ('DSAC-T-Pen', 'C5'),
    'SACFPI': ('SAC-FPI', 'C6'),
    'SACFPIDual': ('SAC-FDPI', 'C3'),
    'CPO': ('CPO', 'C7'),
    'PPOLag': ('PPO-Lag', 'C8'),
}

ALGLABELS = {
    'DSACT': 'DSAC-T',
    'CPO': 'CPO',
    'RCPO': 'RCPO',
    'FOCOPS': 'FOCOPS',
    'CUP': 'CUP',
    'PPOLag': 'PPO-Lag',
    'SACLag': 'SAC-Lag',
    'DSACTPen': 'DSAC-T-Pen',
    'SACFPI': 'SAC-FPI',
    'SACFPIDual': 'SAC-FDPI',
}

ENVTAGRANGES = {
    'SafetyPointGoal1-v0': {
        'cost': (-3, 30),
        'return': (-10, 30),
    },
    'SafetyPointPush1-v0': {
        'cost': (-5, 50),
        'return': (-10, 35),
    },
    'SafetyPointButton1-v0': {
        'cost': (-10, 100),
        'return': (-9, 9),
    },
    'SafetyPointCircle1-v0': {
        'cost': (-8, 80),
    },
    'SafetyCarGoal1-v0': {
        'cost': (-5, 50),
    },
    'SafetyCarPush1-v0': {
        'cost': (-5, 50),
    },
    'SafetyCarButton1-v0': {
        'cost': (-15, 150),
    },
    'SafetyCarCircle1-v0': {
        'cost': (-8, 80),
    },
    'SafetyAntVelocity-v1': {
        'cost': (-0.5, 5),
    },
    'SafetyHalfCheetahVelocity-v1': {
        'cost': (-5, 50),
    },
    'SafetyHopperVelocity-v1': {
        'cost': (-5, 50),
    },
    'SafetyHumanoidVelocity-v1': {
        'cost': (-0.2, 2),
    },
    'SafetyWalker2dVelocity-v1': {
        'cost': (-1, 10),
    },
    'SafetySwimmerVelocity-v1': {
        'cost': (-8, 80),
    },
}

COSTMAGNIFIERRANGES = {
    'SafetyPointGoal1-v0': (-0.2, 2),
    'SafetyPointPush1-v0': (-0.3, 3),
    'SafetyCarGoal1-v0': (-0.3, 3),
    'SafetyCarPush1-v0': (-0.4, 4),
    'SafetyPointCircle1-v0': (-0.5, 5),
    'SafetyCarCircle1-v0': (-0.5, 5),
    'SafetyAntVelocity-v1': (-0.04, 0.4),
    'SafetyPointButton1-v0': (-0.5, 5),
    'SafetyCarButton1-v0': (-1, 10),
    'SafetyHalfCheetahVelocity-v1': (-0.3, 3),
    'SafetyHopperVelocity-v1': (-0.2, 2),
    'SafetyHumanoidVelocity-v1': (-0.01, 0.1),
    'SafetyWalker2dVelocity-v1': (-0.04, 0.4),
    'SafetySwimmerVelocity-v1': (-0.2, 2),
}

ENVTITLES = {
    'SafetyPointGoal1-v0': 'PointGoal',
    'SafetyPointPush1-v0': 'PointPush',
    'SafetyPointCircle1-v0': 'PointCircle',
    'SafetyCarGoal1-v0': 'CarGoal',
    'SafetyCarPush1-v0': 'CarPush',
    'SafetyCarCircle1-v0': 'CarCircle',
    'SafetyAntVelocity-v1': 'AntVelocity',
    'SafetyHumanoidVelocity-v1': 'HumanoidVelocity',
    'SafetyPointButton1-v0': 'PointButton',
    'SafetyCarButton1-v0': 'CarButton',
    'SafetyHalfCheetahVelocity-v1': 'HalfCheetahVelocity',
    'SafetyHopperVelocity-v1': 'HopperVelocity',
    'SafetyWalker2dVelocity-v1': 'Walker2dVelocity',
    'SafetySwimmerVelocity-v1': 'SwimmerVelocity',
    'SafetyHopper-v4': 'Hopper',
}

ENVTAGLEFTMARGINS = {
    'SafetyAntVelocity-v1': {
        'return': 0.15,
    },
    'SafetyHalfCheetahVelocity-v1': {
        'return': 0.15,
    },
    'SafetyHopperVelocity-v1': {
        'return': 0.15,
    },
    'SafetyHumanoidVelocity-v1': {
        'return': 0.15,
    },
    'SafetyWalker2dVelocity-v1': {
        'return': 0.15,
    },
}

NOMAGNIFIERENVS = []

def extract_tensorboard_data(envs: Sequence[str], algs: Sequence[str], tags: Sequence[str]):
    for env in envs:
        env_dir = os.path.join(LOG_PATH, env)
        for log_dir_name in os.listdir(env_dir):
            idx = log_dir_name.find('_seed')
            alg = log_dir_name[:idx]
            if alg not in algs:
                continue
            seed = log_dir_name[idx + 5:].split('_')[0]
            log_dir = os.path.join(env_dir, log_dir_name)
            for log_file_name in os.listdir(log_dir):
                if log_file_name.startswith('events.out.tfevents'):
                    tb_file = os.path.join(log_dir, log_file_name)
                    break
            ea = event_accumulator.EventAccumulator(tb_file)
            ea.Reload()
            for tag in tags:
                step = []
                value = []
                for event in ea.Scalars(TBTAGS[tag]):
                    step.append(event.step)
                    value.append(event.value)
                df = pd.DataFrame({
                    'step': step,
                    'value': value,
                })
                result_dir = os.path.join(RESULT_PATH, env, tag)
                os.makedirs(result_dir, exist_ok=True)
                result_file_name = '_'.join([alg, seed]) + '.csv'
                result_file = os.path.join(result_dir, result_file_name)
                df.to_csv(result_file, index=False)


def plot_training_curve(envs: Sequence[str], algs: Sequence[str], tags: Sequence[str], step: np.ndarray, magnify_last: float = 0.1):
    save_dir = os.path.join(FIGURE_PATH, 'training_curve')
    os.makedirs(save_dir, exist_ok=True)
    m = int(len(step) * magnify_last)
    for env in envs:
        for tag in tags:
            dfs = []
            tag_dir = os.path.join(RESULT_PATH, env, tag)
            for tag_file_name in os.listdir(tag_dir):
                alg, seed = tag_file_name.split('.')[0].split('_')
                if alg not in algs:
                    continue
                tag_file = os.path.join(tag_dir, tag_file_name)
                df = pd.read_csv(tag_file)
                dfs.append(pd.DataFrame({
                    'step': step,
                    'value': np.interp(step, df['step'], df['value']),
                    'alg': alg,
                    'seed': seed,
                }))
            df = (
                pd.concat(dfs)
                .groupby(['alg', 'step'])
                .apply(mean_confidence_interval, include_groups=False)
                .reset_index()
            )
            algs = df['alg'].unique()
            algs = [alg for alg in ALGLABELCOLORS.keys() if alg in algs]  # sort
            sns.set_theme(style='dark')
            _, ax = plt.subplots(figsize=(5, 4))
            magnifier = tag == 'cost' and env not in NOMAGNIFIERENVS
            if magnifier:
                ax_inset = inset_axes(ax, width='40%', height='30%', loc='upper right')
                x1, x2 = step[-m], step[-1]
                y1, y2 = COSTMAGNIFIERRANGES[env]
            for alg in algs:
                df_alg = df[df['alg'] == alg]
                mean = df_alg['mean']
                ci = df_alg['ci']
                color = ALGLABELCOLORS[alg][1]
                ax.plot(step, mean, color=color)
                ax.fill_between(step, mean - ci, mean + ci, facecolor=color, alpha=0.2)
                if magnifier:
                    ax_inset.plot(step[-m:], mean[-m:], color=color)
                    ax_inset.fill_between(step[-m:], mean[-m:] - ci[-m:], mean[-m:] + ci[-m:],
                                          facecolor=color, alpha=0.2)
            ax.set_xlim(step[0], step[-1])
            ax.set_title(ENVTITLES[env])
            ax.set_xlabel('Environment step')
            ax.set_ylabel(TAGLABELS[tag])
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if env in ENVTAGRANGES.keys() and tag in ENVTAGRANGES[env].keys():
                ax.set_ylim(ENVTAGRANGES[env][tag])
            ax.yaxis.set_major_locator(MaxNLocator(6))
            ax.grid()
            if magnifier:
                ax_inset.set_xlim(x1, x2)
                ax_inset.set_ylim(y1, y2)
                ax_inset.grid()
                rect = Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='black', facecolor='none',
                                zorder=10, clip_on=False)
                ax.add_patch(rect)
                con1 = ConnectionPatch(
                    xyA=(x1, y2), xyB=(0, 0),
                    coordsA='data', coordsB='axes fraction',
                    axesA=ax, axesB=ax_inset,
                    linestyle='--', color='black', alpha=0.5,
                    zorder=10, clip_on=False,
                )
                ax.add_artist(con1)
                con2 = ConnectionPatch(
                    xyA=(x2, y2), xyB=(1, 0),
                    coordsA='data', coordsB='axes fraction',
                    axesA=ax, axesB=ax_inset,
                    linestyle='--', color='black', alpha=0.5,
                    zorder=10, clip_on=False,
                )
                ax.add_artist(con2)
            if env in ENVTAGLEFTMARGINS.keys() and tag in ENVTAGLEFTMARGINS[env].keys():
                left = ENVTAGLEFTMARGINS[env][tag]
            else:
                left = 0.13
            plt.subplots_adjust(left=left, bottom=0.15, right=0.95, top=0.92)
            plt.savefig(os.path.join(save_dir, f'{env}_{tag}.{EXTENSION}'), dpi=300)
            plt.close()


def plot_legend():
    sns.set_theme(style='dark')
    plt.figure(figsize=(10, 1))
    legend_elements = [
        Line2D([0], [0], color=color, lw=2, label=alg)
        for alg, color in ALGLABELCOLORS.values()
    ]
    plt.legend(handles=legend_elements, loc='center', ncol=len(legend_elements),
               handlelength=1, frameon=False)
    plt.axis('off')
    save_dir = os.path.join(FIGURE_PATH, 'training_curve')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'legend.{EXTENSION}'), dpi=300)
    plt.close()


def get_statistics(
    envs: Sequence[str],
    algs: Sequence[str],
    tags: Sequence[str],
    *,
    last: Union[float, Sequence[float]] = 0.1,
    save: bool = True,
):
    data = []
    if not isinstance(last, Sequence):
        last = [last] * len(tags)
    for env in envs:
        for tag, l in zip(tags, last):
            tag_dir = os.path.join(RESULT_PATH, env, tag)
            for tag_fn in os.listdir(tag_dir):
                alg, seed = tag_fn.rsplit('.', 1)[0].rsplit('_', 1)
                if alg not in algs:
                    continue
                tag_file = os.path.join(tag_dir, tag_fn)
                df = pd.read_csv(tag_file)
                last_n = int(len(df) * l)
                data.append({
                    'env': env,
                    'alg': alg,
                    'tag': tag,
                    'value': np.mean(df['value'][-last_n:]),
                    'seed': seed,
                })
    keys = data[0].keys()
    data = {k: [d[k] for d in data] for k in keys}
    df = (
        pd.DataFrame(data)
        .groupby(['env', 'alg', 'tag'])
        .apply(mean_confidence_interval, include_groups=False)
        .reset_index()
    )
    if save:
        os.makedirs(RESULT_PATH, exist_ok=True)
        df.to_csv(os.path.join(RESULT_PATH, 'statistics.csv'), float_format='%.2f', index=False)
    return df


def get_table():
    df = pd.read_csv(os.path.join(RESULT_PATH, 'statistics.csv'))
    envs = sorted(df['env'].unique())
    algs = ALGLABELS.keys()

    pivot_df = df.pivot_table(index=['alg', 'env'], columns='tag', values=['mean', 'ci'])

    pivot_df.columns = [f'{col[1]}_{col[0]}' for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    latex_table = r'''\begin{table}[ht]
    \centering
    \caption{Average cost and return in the last 10\% iterations}
    \resizebox{\textwidth}{!}{
    \begin{tabular}{lcccccc}
        \toprule
'''
    tab = '    '

    for i in range(0, len(envs), 3):
        group_envs = envs[i:min(i + 3, len(envs))]

        latex_table += tab * 2
        for env in group_envs:
            latex_table += f' & \multicolumn{{2}}{{c}}{{{ENVTITLES[env]}}}'
        latex_table += ' \\\\\n'

        latex_table += tab * 2
        for j in range(len(group_envs)):
            latex_table += f'\cmidrule(lr){{{j * 2 + 2}-{j * 2 + 3}}} '
        latex_table += '\n'

        latex_table += tab * 2 + 'Algorithm'
        latex_table += ' & Cost & Return' * len(group_envs)
        latex_table += ' \\\\\n'
        latex_table += tab * 2 + '\midrule\n'

        for alg in algs:
            latex_table += tab * 2 + ALGLABELS[alg]
            for env in group_envs:
                cost_row = pivot_df[(pivot_df['alg'] == alg) & (pivot_df['env'] == env)]
                cost_mean = cost_row['cost_mean'].values[0]
                cost_ci = cost_row['cost_ci'].values[0]
                cost_str = f'${cost_mean:.2f}\\pm{cost_ci:.2f}$'

                ret_row = pivot_df[(pivot_df['alg'] == alg) & (pivot_df['env'] == env)]
                ret_mean = ret_row['return_mean'].values[0]
                ret_ci = ret_row['return_ci'].values[0]
                ret_str = f'${ret_mean:.2f}\\pm{ret_ci:.2f}$'

                latex_table += f' & {cost_str} & {ret_str}'
            latex_table += ' \\\\\n'

        if i + 3 < len(envs):
            latex_table += tab * 2 + '\midrule\n'

    latex_table += r'''        \bottomrule
    \end{tabular}
    }
\end{table}'''

    with open('table.tex', 'w') as f:
        f.write(latex_table)


def plot_violation_ratio(envs: List[str], algs: List[str], last: float = 0.1):
    assert 'SACFPIDual' in algs

    if algs[0] != 'SACFPIDual':
        # move SACFPIDual to the first
        algs.insert(0, algs.pop(algs.index('SACFPIDual')))

    data = {alg: {'mean': [], 'ci': []} for alg in algs}
    for env in envs:
        tag_dir = os.path.join(RESULT_PATH, env, 'violation')
        values = {alg: [] for alg in algs}
        for tag_file_name in os.listdir(tag_dir):
            alg = tag_file_name.split('.')[0].split('_')[0]
            if alg not in algs:
                continue
            tag_file = os.path.join(tag_dir, tag_file_name)
            df = pd.read_csv(tag_file)
            last_n = int(len(df) * last)
            values[alg].append(np.mean(df['value'][-last_n:]))
        for alg in algs:
            data[alg]['mean'].append(np.mean(values[alg]))
            data[alg]['ci'].append(1.96 * np.std(values[alg]) / np.sqrt(len(values[alg])))

    index = np.argsort(data['SACFPIDual']['mean'])[::-1]
    for alg in algs:
        for k, v in data[alg].items():
            data[alg][k] = np.asarray(v)[index]

    sns.set_theme(style='dark')
    _, ax = plt.subplots(figsize=(10, 4))
    bar_width = 0.4
    x = np.arange(len(envs))
    for i, alg in enumerate(algs):
        offset = bar_width * i
        ax.bar(x + offset, data[alg]['mean'], bar_width, color=f'C{1 - i}', label=ALGLABELCOLORS[alg][0])
        ax.errorbar(x + offset, data[alg]['mean'], yerr=data[alg]['ci'], fmt='none', ecolor='black', capsize=4)
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels([ENVTITLES[envs[i]] for i in index], rotation=45, ha='right')
    ax.legend()
    ax.set_ylabel('Violation sample ratio')
    ax.grid()
    plt.tight_layout()
    os.makedirs(FIGURE_PATH, exist_ok=True)
    plt.savefig(os.path.join(FIGURE_PATH, f'violation_ratio.{EXTENSION}'), dpi=300)
    plt.close()


def mean_confidence_interval(group):
    mean = group['value'].mean()
    std = group['value'].std()
    n = group['seed'].nunique()
    ci = 1.96 * std / np.sqrt(n) if n > 1 else 0  # 0.95 confidence interval
    return pd.Series({'mean': mean, 'ci': ci})


def get_agent(env, config):
    if config['alg'] == 'SACFPI':
        agent = SACFPIAgent(
            key=jax.random.PRNGKey(0),
            obs_dim=env.observation_space.shape[-1],
            act_dim=env.action_space.shape[-1],
            hidden_sizes=[config['hidden_dim']] * config['hidden_num'],
        )
    elif config['alg'] == 'SACFPIDual':
        agent = SACFPIDualAgent(
            key=jax.random.PRNGKey(0),
            obs_dim=env.observation_space.shape[-1],
            act_dim=env.action_space.shape[-1],
            hidden_sizes=[config['hidden_dim']] * config['hidden_num'],
        )
    return agent


def sample_data(log_dir: str, param_step: int = 995001, sample_num: int = 100000,
                env_num: int = 10, cost_gamma: float = 0.97, seed: int = 0):
    log_dir = os.path.join(PROJECT_ROOT, log_dir)
    with open(os.path.join(log_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    env = make_env(config['env'], env_num=env_num)
    agent = get_agent(env, config)
    agent.load(os.path.join(log_dir, f'params_{param_step}.pkl'))

    data = {
        'obs': [],
        'action': [],
        'g': [],
    }
    ep_data = {
        'obs': [[] for _ in range(env_num)],
        'action': [[] for _ in range(env_num)],
        'cost': [[] for _ in range(env_num)],
    }
    step = 0
    obs, _ = env.reset(seed=seed)
    while step < sample_num:
        action = agent.get_deterministic_action(obs)
        next_obs, _, cost, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        for i in range(env_num):
            ep_data['obs'][i].append(obs[i])
            ep_data['action'][i].append(action[i])
            ep_data['cost'][i].append(cost[i])
            if done[i]:
                ep_g = [ep_data['cost'][i][-1]]
                for c in ep_data['cost'][i][-2::-1]:
                    ep_g.append(c + (1 - c) * cost_gamma * ep_g[-1])
                ep_g = ep_g[::-1]
                data['obs'].extend(ep_data['obs'][i])
                data['action'].extend(ep_data['action'][i])
                data['g'].extend(ep_g)
                step += len(ep_g)
                if step % 10000 == 0:
                    print(f'Sampled {step}/{sample_num} data')
                for k in ep_data.keys():
                    ep_data[k][i] = []
        obs = next_obs
    for k, v in data.items():
        data[k] = np.stack(v)
    np.savez(os.path.join(log_dir, 'normal_data.npz'), **data)


def plot_g_error(log_dirs: Sequence[str], param_step: int = 995001, bin_num: int = 20):
    sns.set_theme(style='dark')
    _, ax = plt.subplots(figsize=(5, 4))

    errors = {}
    for log_dir in log_dirs:
        log_dir = os.path.join(PROJECT_ROOT, log_dir)
        data = dict(np.load(os.path.join(log_dir, 'normal_data.npz')))

        with open(os.path.join(log_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        env = make_env(config['env'])
        agent = get_agent(env, config)
        agent.load(os.path.join(log_dir, f'params_{param_step}.pkl'))

        obs = data['obs']
        action = data['action']
        gtg = data['g']
        g = np.maximum(
            agent.scenery(agent.params.g1, obs, action),
            agent.scenery(agent.params.g2, obs, action),
        )
        errors[config['alg']] = g - gtg

    min_val = min([e.min() for e in errors.values()])
    max_val = max([e.max() for e in errors.values()])
    bins = np.linspace(min_val, max_val, bin_num)
    for alg, error in errors.items():
        ax.hist(error, bins=bins, log=True, label=ALGLABELCOLORS[alg][0], alpha=0.8)
    ax.set_xlabel('G error')
    ax.set_ylabel('Frequency')
    ax.set_title(ENVTITLES[config['env']])
    ax.legend()
    ax.grid()
    plt.tight_layout()
    save_dir = os.path.join(FIGURE_PATH, 'g_error')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{config["env"]}_g_error.{EXTENSION}'), dpi=300)
    plt.close()


def get_snapshot(envs: Sequence[str], width: int = 1024, height: int = 1024):
    save_path = os.path.join(FIGURE_PATH, 'snapshot')
    os.makedirs(save_path, exist_ok=True)
    for env_id in envs:
        if 'Velocity' in env_id:
            camera_id = 0
        else:
            camera_id = 1
        env = make_env(env_id, render_mode='rgb_array',
                       width=width, height=height, camera_id=camera_id)
        env.reset()
        if 'Velocity' in env_id:
            for _ in range(10):
                env.step(env.action_space.sample())
        frame = env.render()
        plt.imsave(os.path.join(save_path, env_id + '.png'), frame)
        env.close()


def plot_cost_return_scatter(envs: Sequence[str], algs: Sequence[str], normalize_by='FOCOPS'):
    stats_file = os.path.join(RESULT_PATH, 'statistics.csv')
    if not os.path.exists(stats_file):
        print(f"Statistics file {stats_file} not found. Please run get_statistics() first.")
        return
    
    df = pd.read_csv(stats_file)
    df = df[df['env'].isin(envs) & df['alg'].isin(algs)]

    for env in envs:
        for tag in ['cost', 'return']:
            if not ((df['env'] == env) & (df['alg'] == normalize_by) & (df['tag'] == tag)).any():
                assert False, f"Missing {normalize_by} data for {env} in {tag}."
                
            baseline = df.loc[(df['env'] == env) & (df['alg'] == normalize_by) & (df['tag'] == tag), 'mean'].values[0]
            if baseline == 0:
                assert False, f"Baseline value for {normalize_by} in {env} is zero."
                
            df.loc[(df['env'] == env) & (df['tag'] == tag), 'mean'] /= baseline

    sns.set_theme(style='dark')
    _, ax = plt.subplots(figsize=(8, 5))
    for alg in algs:
        if alg == normalize_by:
            continue
        label, color = ALGLABELCOLORS[alg]
        cost = df.loc[(df['alg'] == alg) & (df['tag'] == 'cost'), 'mean']
        ret = df.loc[(df['alg'] == alg) & (df['tag'] == 'return'), 'mean']
        cost_mean = cost.mean()
        ret_mean = ret.mean()
        cost_ci = 1.96 * cost.std() / len(cost)
        ret_ci = 1.96 * ret.std() / len(ret)
        ax.errorbar(
            cost_mean,
            ret_mean,
            xerr=cost_ci,
            yerr=ret_ci,
            label=label,
            color=color,
            fmt='o',
            markersize=6,
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
        )
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.set_xlabel('Normalized cost')
    ax.set_ylabel('Normalized return')
    ax.xaxis.set_major_locator(LogLocator(subs='all'))  # Shows all decades and sub-decades
    ax.xaxis.set_minor_locator(LogLocator(subs=np.arange(1, 10) * 0.1, numticks=10))
    plt.grid()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, f'return_cost_scatter.{EXTENSION}'), dpi=300)


def normalize_by(df: pd.DataFrame, baseline: str):
    for env in df['env'].unique():
        for tag in df['tag'].unique():
            if not ((df['env'] == env) & (df['alg'] == baseline) & (df['tag'] == tag)).any():
                assert False, f"Missing {baseline} data for {env}/{tag}."

            bv = df.loc[(df['env'] == env) & (df['alg'] == baseline) & (df['tag'] == tag), 'mean'].values[0]
            if bv == 0:
                assert False, f"Baseline value for {baseline} in {env}/{tag} is zero."

            df.loc[(df['env'] == env) & (df['tag'] == tag), 'mean'] /= bv
