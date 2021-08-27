import retro
import torch as th
from retro import Actions
from stable_baselines3 import PPO as PPO


# if you want t0 import your own ROM, firsts please make sure they are already include in the retro.data.game_list
# /,then put it together with ur project


# class CustomReward(Wrapper):
#     def __init__(self, env=None):
#         super(CustomReward, self).__init__(env)
#         self.observation_space = env.observation_space
#         self.current_position = 0
#         self.current_score = 0
#         self.curr_lives = 2
#
#     def step(self, action):
#         state, _, done, info = self.env.step(action)


class game:

    def __init__(self, name, time_steps, level):
        self.game_name = name
        self.env = retro.make(name, state=level, use_restricted_actions=Actions.DISCRETE)
        self.total_time_steps = time_steps

    def __call__(self):
        policy_kwargs = dict(activation_fn=th.nn.ReLU)
        model = PPO('CnnPolicy',
                    self.env,
                    learning_rate=1e-3,
                    policy_kwargs=policy_kwargs).learn(self.total_time_steps)
        model.save('PPO_' + self.game_name)
        del model   # since the model has been trained, its no longer needed any more...


if __name__ == '__main__':
    g = game('Contra-Nes', 3000, 'Level1')
    g()
