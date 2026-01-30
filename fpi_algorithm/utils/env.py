import safety_gymnasium
from gymnasium import Env
from safety_gymnasium.wrappers import SafeRescaleAction
from safety_gymnasium.vector.async_vector_env import SafetyAsyncVectorEnv
from fpi_algorithm.utils.wrapper import BinaryCostWrapper, RewardScalingWrapper


def make_env(env_id: str, env_num: int = 1, train: bool = False, **kwargs) -> Env:
    def env_fn() -> Env:
        env = safety_gymnasium.make(env_id, **kwargs)
        env = SafeRescaleAction(env, -1.0, 1.0)
        if train:
            env = BinaryCostWrapper(env)
            env = RewardScalingWrapper(env, reward_scale(env_id))
        return env
    if env_num == 1:
        env = env_fn()
    else:
        env = SafetyAsyncVectorEnv([env_fn for _ in range(env_num)])
    return env


def reward_scale(env_id: str) -> float:
    if 'Velocity' in env_id and 'Swimmer' not in env_id:
        scale = 0.01
    else:
        scale = 1.0
    return scale
