import gym
from gym.wrappers import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from datetime import datetime
import signal

now = datetime.now()
dt_string = now.strftime("%y-%m-%d %H:%M:%S")

def wrap_env(env):
    env = Monitor(env, 'video/', force=True)
    return env

log_path = f"logs/{dt_string}"
logger = configure(log_path, ["stdout", "tensorboard"])

env = gym.make('Breakout-v0') # or Breakout-v0 for Atari
env = wrap_env(env)

model = PPO('MlpPolicy', env, verbose=1)
model.set_logger(logger)
model.learn(total_timesteps=10000000)

obs = env.reset()
for i in range(100000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()


def handle_signal(sig, frame):
    print("Exiting gracefully, saving model")
    model.save(f"{dt_string}.pkl")


signal.signal(signal.SIGINT, handle_signal)

model.save(f"{dt_string}.pkl")
