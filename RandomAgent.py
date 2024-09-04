# RandomAgent.py
from gym.wrappers import RecordVideo
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import warnings
warnings.filterwarnings('ignore')

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = RecordVideo(env, 'video', episode_trigger=lambda x: x == 2)
env.start_video_recorder()

obs = env.reset()
done = False
score = 0
while not done:
    env.render()
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    score += rew
env.close()
