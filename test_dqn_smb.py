# test_dqn_atari.py
# This script tests a model for Super Mario Bros 1-1
import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import multiprocessing
import time
import dqn
import action_space
import control
import sys

from dqn_utils import *
from gym.envs.classic_control import rendering

# Acquire all available gpus from the local device
def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

# Sets global random seeds
def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i) 
    np.random.seed(i)
    random.seed(i)

# Gets the current session
def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session

# Gets the gym environment
# Parameter: task - the gym environment id to load
# Parameter: seed – the environment seed to be set
def get_env(task, seed):
    env_id = task

    env = gym.make(env_id)
    multi_lock = multiprocessing.Lock() # as described in ppaquette's gym smb github page
    env._configure(lock = multi_lock)
    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/smb_vid_dir2/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    wrapper = action_space.ToDiscrete() # Convert the action_space from multidiscrete to discrete to allow for appropriate actions for smb
    env = wrapper(env)
    
    return env

# The main function to test the progress of the model (based off of dqn.py)
# Parameter: model_name - the name of the model to test
# Parameter: env - the gym environment to load
# Parameter: replay_buffer_size - the max number of transitions to store in the buffer. When the buffer overflows the old memories are dropped.
# Parameter: frame_history_len - the number of memories to be retried for each observation
def test_model(model_name, env, replay_buffer_size = 100000, frame_history_len = 4):

    with tf.Session(graph = tf.Graph()) as sess:

        # Load in the model from most recent checkpoint
        graph_name = model_name + '.meta'
        saver = tf.train.import_meta_graph(graph_name)
        saver.restore(sess, tf.train.latest_checkpoint('./'))   # Requires the checkpoint file to contain the model loaded in with model_name
        meta_graph = tf.get_default_graph()


        if len(env.observation_space.shape) == 1:
            # This means we are running on low-dimensional observations (e.g. RAM)
            input_shape = env.observation_space.shape
        else:
            img_h, img_w, img_c = env.observation_space.shape
            input_shape = (img_h, img_w, frame_history_len * img_c)

        num_actions = env.action_space.n

        # set up placeholders
        # placeholder for current observation (or state)
        obs_t_ph              = meta_graph.get_tensor_by_name('obs_t_ph:0') 
        # placeholder for current action   
        act_t_ph              = meta_graph.get_tensor_by_name('act_t_ph:0') 
        # placeholder for current reward
        rew_t_ph              = meta_graph.get_tensor_by_name('rew_t_ph:0')
        # placeholder for next observation (or state)
        obs_tp1_ph            = meta_graph.get_tensor_by_name('obs_tp1_ph:0')
        # placeholder for end of episode mask
        # this value is 1 if the next state corresponds to the end of an episode,
        # in which case there is no Q-value at the next state; at the end of an
        # episode, only the current state reward contributes to the target, not the
        # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
        done_mask_ph          = meta_graph.get_tensor_by_name('done_mask_ph:0')

        # casting to float on GPU ensures lower data transfer times.
        obs_t_float   = tf.cast(obs_t_ph,   tf.float32) / 255.0
        obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0
        
        # Load current_q_func from model
        current_qfunc = meta_graph.get_tensor_by_name('current_qfunc/current_q_func_op:0')
        # Load output_layer of current_qfunc from model
        output_layer = meta_graph.get_tensor_by_name('current_qfunc/q_func/action_value/fully_connected_1/BiasAdd:0')

        # Redefine action_predict op 
        action_predict = tf.argmax(current_qfunc, axis = 1, name = 'action_predict')
        
        #######################
        # Set up for model test
        #######################

        last_obs = env.reset()
        previous_obs = last_obs # test to see if observations are changing for debug purposes
        replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

        done = False # boolean that determines if finished the game

        mean_episode_reward      = -float('nan')
        best_mean_episode_reward = -float('inf')
        iteration = 0

        random_action_counter = 0   # record of random actions done
        repeat_action_timer = 0     # record of how many times an action was repeated
        last_action = 0             # record of last action used
        last_distance = 0           # record of distance covered
        stuck = False               # boolean to see if stuck in the level
        

        ##################
        # Begin model test
        ##################
        while not done:
            obs_idx = replay_buffer.store_frame(last_obs)
            replay_obs = replay_buffer.encode_recent_observation()
            replay_obs = replay_obs.reshape(1,replay_obs.shape[0],replay_obs.shape[1],replay_obs.shape[2])  # reshape replay_obs for processing later
            

            if random_action_counter < 0 or stuck: # random actions are never done during testing, so this statement will run only if agent is stuck
                if stuck:
                    stuck = False
                    action = env.action_space.sample()  # if agent is stuck, take a random action (often happens when agent is stuck against a pipe or block)
                else:
                    action = env.action_space.sample()
                
                last_action = action
                last_obs, reward, done, info = env.step(action)
                
                previous_obs = last_obs
                random_action_counter += 1

            else:                                  # not stuck, so will proceed with model's prediction
                if repeat_action_timer >= 0:

                    repeat_action_timer = 0
                    action = sess.run([action_predict], feed_dict = {obs_t_ph: replay_obs})[0]
                    last_action = action

                    # Due to reshaping issues, this if statement was necessary
                    # The action is still predicted by the model
                    if not isinstance(action,int):
                        last_obs, reward, done, info = env.step(action[0])

                    else:
                        last_obs, reward, done, info = env.step(action)

                    previous_obs = last_obs    

                else:

                    repeat_action_timer += 1
                    action = last_action

                    if not isinstance(action,int):
                        last_obs, reward, done, info = env.step(action[0])

                    else:
                        last_obs, reward, done, info = env.step(action)

                    previous_obs = last_obs
              


            if iteration % 25 == 0: # distance check to make sure mario has moved

                curr_distance = info['distance']    # Acquire distance covered by agent
                if curr_distance == last_distance:
                    stuck = True # Easy way to make agent take another action on the next timestep since he got stuck
                    # print 'STUCK CODE RUN'
                
                last_distance = curr_distance
            
            replay_buffer.store_effect(obs_idx, action, reward, done)


            
            # Log reward info
            episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
            if len(episode_rewards) > 0:
                mean_episode_reward = np.mean(episode_rewards[-100:])
            if len(episode_rewards) > 100:
                best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
            if done:
                print("Iteration %d" % (iteration,))
                print("mean reward (100 episodes) %f" % mean_episode_reward)
                print("best mean reward %f" % best_mean_episode_reward)
                print("episodes %d" % len(episode_rewards))
                # print("exploration %f" % exploration.value(t))
                # print("learning_rate %f" % optimizer_spec.lr_schedule.value(t))
                sys.stdout.flush()
                

            iteration += 1

# Main that gets the environment and the session before testing begins
def main():

    task = 'SuperMarioBros-1-1-v0'  # The environment to load
    seed = 0                        # Arbitrarily choose seed of 0
    env = get_env(task, seed)
    session = get_session()
    tf.reset_default_graph()

    # Test the model
    test_model('SMB_model-241', env)

if __name__ == "__main__":
    main()
