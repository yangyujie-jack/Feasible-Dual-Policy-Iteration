import numpy as np
from fpi_algorithm.utils.random import seeding
from fpi_algorithm.utils.experience import Experience


class Buffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int, seed: int = 0):
        self.buffer = (
            np.zeros((size, obs_dim), dtype=np.float32),  # obs
            np.zeros((size, act_dim), dtype=np.float32),  # action
            np.zeros((size, obs_dim), dtype=np.float32),  # next_obs
            np.zeros((size,), dtype=np.float32),          # reward
            np.zeros((size,), dtype=np.float32),          # cost
            np.zeros((size,), dtype=bool),                # done
        )
        self.size = size
        self.ptr = 0
        self.len = 0
        self.rng, _ = seeding(seed)

    def __len__(self):
        return self.len

    def add(self, *args):
        for buf, elem in zip(self.buffer, args):
            buf[self.ptr] = elem
        self.ptr = (self.ptr + 1) % self.size
        self.len = min(self.len + 1, self.size)

    def add_batch(self, *args):
        batch_size = args[0].shape[0]
        start, end = self.ptr, self.ptr + batch_size
        if end > self.size:
            split, remain = self.size - start, end - self.size
            for buf, elem in zip(self.buffer, args):
                buf[start:] = elem[:split]
                buf[:remain] = elem[split:]
        else:
            for buf, elem in zip(self.buffer, args):
                buf[start:end] = elem
        self.ptr = (self.ptr + batch_size) % self.size
        self.len = min(self.len + batch_size, self.size)

    def sample(self, size: int) -> Experience:
        idxes = self.rng.integers(0, self.len, size=size)
        return Experience(*[buf[idxes] for buf in self.buffer])
