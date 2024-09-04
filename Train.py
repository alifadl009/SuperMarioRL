# Train.py
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

callback = TrainAndLoggingCallback(check_freq=100000, save_path=check_dir)

# Training
learning_rate=0.0001
n_step = 512

model = PPO('CnnPolicy', env=env, verbose=1, tensorboard_log=log_dir, learning_rate=learning_rate)
model.learn(total_timesteps=1000000, callback=callback)

lr_str = '00001'
model.save('model_' + lr_str)
env.close()

