import retro
from retro import Actions
from stable_baselines3 import PPO as PPO
# import retro_test


model = PPO.load('PPO_' + 'ContraForce-Nes')
env = retro.make(game='ContraForce-Nes', use_restricted_actions=Actions.DISCRETE)
obs = env.reset()
for i in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
