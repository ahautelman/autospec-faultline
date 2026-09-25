from env.dirl_grid import RoomsEnv
from env.rooms_envs import GRID_PARAMS_LIST, MAX_TIMESTEPS, START_ROOM, FINAL_ROOM
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.logger import configure


tmp_path = "/tmp/sb3_log/"
# set up logger
new_logger = configure(tmp_path, ["stdout"])


env = Monitor(RoomsEnv(GRID_PARAMS_LIST[0], START_ROOM[0], FINAL_ROOM[0]))
env = Monitor(env, info_keywords=("is_success",))
model = PPO('MlpPolicy', env, verbose = 1)

model.learn(100000)
model.set_logger(new_logger)

vec_env = model.get_env()
obs = vec_env.reset()
rewards = 0
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    rewards += reward

print(rewards)