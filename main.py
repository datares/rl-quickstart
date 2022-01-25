import gym
from gym.wrappers import Monitor
from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure

def wrap_env(env):
    env = Monitor(env, 'video/', force=True)
    return env

log_path = "logs/"
# # set up logger
logger = configure(log_path, ["stdout", "tensorboard"])

env = gym.make('CartPole-v1') # Breakout-v0
env = wrap_env(env)

model = A2C('MlpPolicy', env, verbose=1)
model.set_logger(logger)
model.learn(total_timesteps=10000, tb_log_name="first run")

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()
