# import gym
# import gym_pull
# gym_pull.pull('github.com/ppaquette/gym-super-mario')        # Only required once, envs will be loaded with import gym_pull afterwards
# env = gym.make('ppaquette/SuperMarioBros-1-1-v0')

import gym
import time
from dqn_utils import *
import multiprocessing
# from .action_space import *
# from .control import *

env = gym.make('SuperMarioBros-1-1-v0')
print type(env)
multi_lock = multiprocessing.Lock()
env._configure(lock = multi_lock)
# wrapper = control.SetPlayingMode('human')
# env = wrapper(env)
observation = env.reset()
print env.action_space.shape

# print get_wrapper_by_name(env, "Monitor").get_total_steps()
for _ in xrange(5):
	action = env.action_space.sample()
	print action
	time.sleep(1)
	obs, rew, done, info = env.step(action)
	# env.render()
env._close()
env.reset()
action = env.action_space.sample()
env.step(action)
# env.render()
# for _ in range(1000):
#     env.render()
#     action = env.action_space.sample() # your agent here (this takes random actions)
#     observation, reward, done, info = env.step(action)