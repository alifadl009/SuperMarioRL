# ReTrain.py
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation 

from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO
from CallBack import TrainAndLoggingCallback

import warnings
warnings.filterwarnings('ignore')

log_dir = './log'
check_dir = './Training'

# Preprocessing
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda : env])
env = VecFrameStack(env, 4, channels_order='last')

callback = TrainAndLoggingCallback(check_freq=50_000, save_path=check_dir)

# Training

model = PPO.load(path='./Training/best_model_100000', env=env)
model.learn(total_timesteps=1_000_000, callback=callback)


model.save('Training/')
env.close()

