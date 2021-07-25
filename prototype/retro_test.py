import random
import retro
import torch
from torch.nn import functional as f
from torch import nn, optim


class QNet(nn.Sequential):
    def __init__(self, num_inputs, num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, (3, 3), (2, 2), (1, 1))
        self.conv2 = nn.Conv2d(32, 64, (3, 3), (2, 2), (1, 1))
        self.conv3 = nn.Conv2d(64, 64, (3, 3), (2, 2), (1, 1))
        self.conv4 = nn.Conv2d(64, 64, (3, 3), (2, 2), (1, 1))
        self.linear = nn.Linear(64 * 6 * 6, 512)
        self.flatten = nn.Flatten()
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))
        x = f.relu(self.conv4(x))
        x = self.flatten(x)
        x = self.linear(x)
        # 定义 动作模型 和 评估模型
        return self.actor_linear(x), self.critic_linear(x)


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
            state = self.env.reset()
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
                        _state = torch.tensor(state, dtype=torch.float32)
                        Qs = self.q_net(_state[None, ...])
                        action = torch.argmax(Qs, 1)[0].item()
                else:
                    action = self.env.action_space.sample()

                next_state, reward, done, _ = self.env.step(action)
                R += reward
                self.exp_pool.append([state, reward, action, next_state, done])
                state = next_state

                if done:
                    avg = 0.95 * avg + 0.05 * R
                    print(avg, R)
                    if avg > 400:
                        is_render = True
                    break
                    # 训练
                    if len(self.exp_pool) >= self.exp_pool_size:
                        exps = random.choices(self.exp_pool, k=100)
                        _state = torch.tensor([exp[0] for exp in exps]).float()
                        _reward = torch.tensor([[exp[1]] for exp in exps])
                        _action = torch.tensor([[exp[2]] for exp in exps])
                        _next_state = torch.tensor([exp[3] for exp in exps]).float()
                        _done = torch.tensor([[int(exp[4])] for exp in exps])

                        # 预测值
                        _Qs = self.q_net(_state)
                        _Q = torch.gather(_Qs, 1, _action)  # gather，沿dim指定的轴收集值
                        # 目标值
                        _next_Qs = self.q_net(_next_state)
                        _max_Q = torch.max(_next_Qs, dim=1, keepdim=True)[0]
                        _target_Q = _reward + (1 - _done) * 0.9 * _max_Q

                        loss = self.loss_fn(_Q, _target_Q.detach())
                        self.opt.zero_grad()
                        loss.backward()
                        self.opt.step()


if __name__ == '__main__':
    g = game(100, 0.9)
    g()
