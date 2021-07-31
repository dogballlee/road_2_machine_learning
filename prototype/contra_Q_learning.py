import retro
import torch
from torch import nn, optim
import torch.nn.functional as f
from collections import OrderedDict


class Q_net(nn.Sequential):
    super().__init__(
        nn.Linear(4, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 5)
    )


# python -m retro.import D:\py_project\DRL_DEMO\ROMs\
# retro.make('ContraForce-Nes.nes')

class game:
    def __init__(self, exp_pool_size, explore):
        self.env = retro.make(game='ContraForce-Nes', state='Level1')
        self.env.reset()
        self.exp_pool = []
        self.exp_pool_size = exp_pool_size
