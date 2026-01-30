import numpy as np
import matplotlib.pyplot as plt
from fpi_algorithm.utils.plot import extract_tensorboard_data, plot_training_curve, plot_legend, \
    get_statistics, get_table, plot_violation_ratio, sample_data, plot_g_error, get_snapshot, plot_cost_return_scatter


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

envs = [
    'SafetyPointGoal1-v0',
    'SafetyPointPush1-v0',
    'SafetyPointCircle1-v0',
    'SafetyCarGoal1-v0',
    'SafetyCarPush1-v0',
    'SafetyCarCircle1-v0',
    'SafetyAntVelocity-v1',
    'SafetyHumanoidVelocity-v1',
    'SafetyPointButton1-v0',
    'SafetyCarButton1-v0',
    'SafetyHalfCheetahVelocity-v1',
    'SafetyHopperVelocity-v1',
    'SafetyWalker2dVelocity-v1',
    'SafetySwimmerVelocity-v1',
]

algs = [
    'CPO',
    'RCPO',
    'FOCOPS',
    'CUP',
    'PPOLag',
    'SACLag',
    'DSACTPen',
    'SACFPI',
    'SACFPIDual',
]

tags = [
    'cost',
    'return',
    # 'violation',
]

extract_tensorboard_data(envs, algs, tags)

# step = np.linspace(int(2e4), int(2e6), 100)
# plot_training_curve(envs, algs, tags, step)
# plot_legend()
# get_statistics(envs, algs, tags)
# get_table()

# plot_violation_ratio(envs, algs)

log_dirs = []

# plot_g_error(log_dirs)

# get_snapshot(envs)

# plot_cost_return_scatter(envs, algs)
