import retro
from retro import Actions
from stable_baselines3 import PPO as PPO
# import retro_test

g_name = 'ContraForce-Nes'
env = retro.make(game=g_name, use_restricted_actions=Actions.DISCRETE)
model = PPO.load('PPO_' + g_name, env=env)
obs = env.reset()
for i in range(5000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
