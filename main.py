import gym
from gym.wrappers import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from datetime import datetime

ENVIRONMENT = "FetchPickAndPlace-v1"

now = datetime.now()
dt_string = now.strftime("%y-%m-%d %H:%M:%S")

def wrap_env(env):
    env = Monitor(env, f'video/{ENVIRONMENT}-{dt_string}', force=True)
    return env

log_path = f"logs/{ENVIRONMENT}-{dt_string}"
logger = configure(log_path, ["stdout", "tensorboard"])

env = gym.make(ENVIRONMENT)
env = wrap_env(env)

model = PPO('MlpPolicy', env, verbose=1)
model.set_logger(logger)
model.learn(total_timesteps=10000000)

obs = env.reset()
max_reward = 0
for i in range(100000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if reward > max_reward:
        model.save(f"models/{ENVIRONMENT}-{dt_string}-reward={reward}.pkl")
        print(f"max reward = {reward}")
        max_reward = reward
    if done:
      obs = env.reset()

env.close()
