import retro
from stable_baselines3 import PPO as PPO
# from stable_baselines3.common.env_util import make_vec_env, make_atari_env


class game:

    def __init__(self, name, time_steps, level):
        self.game_name = name
        try:
            self.env = retro.make(name, state=level)
        except NameError:
            print('oops, cant find this name in the retro game list, pls check ur input.')

        self.total_time_steps = time_steps

    def __call__(self):
        model = PPO('MlpPolicy', self.env, verbose=1)
        model.learn(self.total_time_steps)
        model.save('PPO_' + self.game_name)
        del model

        model = PPO.load('PPO_' + self.game_name)

        obs = self.env.reset()
        for i in range(1000):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            self.env.render()


if __name__ == '__main__':
    g = game('Airstriker-Genesis', 8000, 'Level1')
    g()
