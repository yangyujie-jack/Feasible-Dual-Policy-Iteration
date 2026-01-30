import gymnasium as gym


class BinaryCostWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, cost, terminated, truncated, info = self.env.step(action)
        return observation, reward, float(cost > 0), terminated, truncated, info


class RewardScalingWrapper(gym.Wrapper):
    def __init__(self, env, reward_scale: float = 1.0):
        super().__init__(env)
        self.reward_scale = reward_scale

    def step(self, action):
        observation, reward, cost, terminated, truncated, info = self.env.step(action)
        return observation, reward * self.reward_scale, cost, terminated, truncated, info


class NormalizeStepOutputWrapper(gym.Wrapper):
    '''Wrap an environment to normalize its step output to standard Gym interface.'''
    def step(self, action):
        observation, reward, cost, terminated, truncated, info = self.env.step(action)
        info['cost'] = cost
        return observation, reward, terminated, truncated, info
