# Evaluate.py
import gym_super_mario_bros
from stable_baselines3 import PPO
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation 

import warnings
warnings.filterwarnings('ignore')

from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda : env])
env = VecFrameStack(env, 4, channels_order='last')

model = PPO.load('./Training/best_model_50000.zip')

eps = 10 # number of episodes to evaluate

for i in range(1, eps+1):
    state = env.reset()
    done = False
    score = 0
    while not done:
        action, _ = model.predict(state)
        tiva, reward, done, info = env.step(action)
        env.render()
        score =+ reward
        
    print(f"Episode: {i}, Score: {score}")    

env.close() 

