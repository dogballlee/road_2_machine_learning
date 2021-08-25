import retro
from retro import Actions
from stable_baselines3 import PPO as PPO


# if you want t0 import your own ROM, firsts please make sure they are already include in the retro.data.game_list
# /,then put it together with ur project

class game:

    def __init__(self, name, time_steps, level):
        self.game_name = name
        try:
            self.env = retro.make(name, state=level, use_restricted_actions=Actions.DISCRETE)
        except NameError:
            print('oops, cant find this name in the retro game list, pls check ur input.')

        self.total_time_steps = time_steps

    def __call__(self):
        model = PPO('MlpPolicy', self.env, learning_rate=1e-4).learn(self.total_time_steps)
        model.save('PPO_' + self.game_name)
        del model   # the train model is no longer needed any more...

        # model = PPO.load('PPO_' + self.game_name)
        #
        # obs = self.env.reset()
        # for i in range(1000):
        #     action, _state = model.predict(obs, deterministic=True)
        #     obs, reward, done, info = self.env.step(action)
        #     self.env.render()


if __name__ == '__main__':
    g = game('ContraForce-Nes', 30000, 'Level1')
    g()
