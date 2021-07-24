import random
import retro
import torch
from torch import nn, optim


class QNet(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )


class game:
    def __init__(self, exp_pool_size, explore):
        self.env = retro.make(game='Airstriker-Genesis')
        self.exp_pool = []
        self.exp_pool_size = exp_pool_size
        self.q_net = QNet()
        self.explore = explore
        self.loss_fn = nn.MSELoss
        self.opt = optim.Adam(self.q_net.parameters())

    def __call__(self):
        is_render = False
        avg = 0
        while True:
            tate = self.env.reset()
            R = 0
            while True:
                if is_render:
                    self.env.render()
                if len(self.exp_pool) >= self.exp_pool_size:
                    self.exp_pool.pop(0)
                    self.explore += 1e-7
                    if torch.rand(1) > self.explore:
                        action = self.env.action_space.sample()
                    else:



def main():
    env = retro.make(game='Airstriker-Genesis')
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()
    env.close()


if __name__ == '__main__':
    main()
