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
print np.count_nonzero(observation)
# print observation
print env.action_space.shape

# print get_wrapper_by_name(env, "Monitor").get_total_steps()
for i in xrange(1000):
	action = env.action_space.sample()
	# print action
	# time.sleep(1)
	obs, rew, done, info = env.step(action)
	# print obs
	# print 'THE ARRAY IS ', np.count_nonzero(obs)
	if np.array_equal(obs,np.zeros(shape=(224, 256, 3), dtype=np.uint8)):
		print 'EMPTY SCREEN ARRAY'
	if np.array_equal(observation,obs):
		print 'obs are the same'
	else:
		print 'obs are NOT the same'

	observation = obs
	# time.sleep(5)
	if done:
		break
	# env.render()
env._close()
reset_obs = env.reset()
print type(reset_obs)
action = env.action_space.sample()
env.step(action)
# env.render()
# for _ in range(1000):
#     env.render()
#     action = env.action_space.sample() # your agent here (this takes random actions)
#     observation, reward, done, info = env.step(action)