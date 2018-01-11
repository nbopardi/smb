# test_dqn_atari.py
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
# from action_space import *
# from control import *


from dqn_utils import *
# from atari_wrappers import *
from gym.envs.classic_control import rendering

# def atari_model(img_in, num_actions, scope, reuse=False):
#     # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
#     with tf.variable_scope(scope, reuse=reuse):
#         out = img_in
#         with tf.variable_scope("convnet"):
#             # original architecture
#             out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
#             out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
#             out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
#         out = layers.flatten(out)
#         with tf.variable_scope("action_value"):
#             out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
#             out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

#         return out

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i) 
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session

def get_env(task, seed):
    env_id = task

    env = gym.make(env_id)
    multi_lock = multiprocessing.Lock()
    env._configure(lock = multi_lock)
    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/smb_vid_dir2/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    wrapper = action_space.ToDiscrete()
    env = wrapper(env)
    # print type(env.action_space)
    
    return env

def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0: 
        if not err: 
            print "Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l)
            err.append('logged')
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)

def test_model(model_name, env, replay_buffer_size = 100000, frame_history_len = 4):
    with tf.Session(graph = tf.Graph()) as sess:
        graph_name = model_name + '.meta'
        saver = tf.train.import_meta_graph(graph_name)
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        meta_graph = tf.get_default_graph()
        # print graph_name
        print meta_graph

        # for op in meta_graph.get_operations():
        #     print str(op.name)
        
        # assert meta_graph is int 
        if len(env.observation_space.shape) == 1:
            # This means we are running on low-dimensional observations (e.g. RAM)
            input_shape = env.observation_space.shape
        else:
            img_h, img_w, img_c = env.observation_space.shape
            input_shape = (img_h, img_w, frame_history_len * img_c)
        num_actions = env.action_space.n

        # set up placeholders
        # placeholder for current observation (or state)
        obs_t_ph              = meta_graph.get_tensor_by_name('obs_t_ph:0')    # placeholder for current action
        act_t_ph              = meta_graph.get_tensor_by_name('act_t_ph:0')
        # placeholder for current rewar
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
        
        current_qfunc = meta_graph.get_tensor_by_name('current_qfunc/current_q_func_op:0')
        output_layer = meta_graph.get_tensor_by_name('current_qfunc/q_func/action_value/fully_connected_1/BiasAdd:0')

        # current_qfunc = meta_graph.get_tensor_by_name("current_qfunc:0")
        # current_qfunc = q_func(obs_t_float, num_actions, scope = "q_func", reuse = False)
        # action_predict = tf.argmax(current_qfunc, axis = 1)
        # action_predict = meta_graph.get_tensor_by_name('action_predict:0')
        action_predict = tf.argmax(current_qfunc, axis = 1, name = 'action_predict')
        # def predict_action(observation):

        #     return sess.run(action_predict, feed_dict = {obs_t_float: observation})[0]


        # tf.global_variables_initializer()
        last_obs = env.reset()
        previous_obs = last_obs # test to see if observations are changing
        # previous_state = np.array(0) # test to see if emulator is returning different states (observations)
        replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

        done = False

        mean_episode_reward      = -float('nan')
        best_mean_episode_reward = -float('inf')
        iteration = 0

        # viewer = rendering.SimpleImageViewer()
        random_action_counter = 0
        repeat_action_timer = 0
        last_action = 0
        last_distance = 0
        stuck = False
        
        while not done:
            # time.sleep(10)
            obs_idx = replay_buffer.store_frame(last_obs)
            replay_obs = replay_buffer.encode_recent_observation()
            # print replay_obs.shape
            replay_obs = replay_obs.reshape(1,replay_obs.shape[0],replay_obs.shape[1],replay_obs.shape[2])
            # print replay_obs.shape
            # action = predict_action(replay_obs)
            if random_action_counter < 0 or stuck:
                if stuck:
                    stuck = False
                    # action = 8
                    action = env.action_space.sample()
                else:
                    action = env.action_space.sample()
                last_action = action
                # print action
                last_obs, reward, done, info = env.step(action)
                
                # if np.array_equal(previous_obs,last_obs):
                #     print 'obs are the same'
                # else:
                #     print 'obs not the same'
                # if np.array_equal(previous_state,temp_state):
                #     print 'states are the same'
                # else:
                #     print 'states are not the same'
                # time.sleep(5)
                previous_obs = last_obs
                # previous_state = temp_state
                random_action_counter += 1

            else:
                if repeat_action_timer >= 0:
                    # time.sleep(5)
                    repeat_action_timer = 0
                    action = sess.run([action_predict], feed_dict = {obs_t_ph: replay_obs})[0]
                    # print sess.run(output_layer, feed_dict = {obs_t_ph: replay_obs})
                    # print output
                    last_action = action



                    # print 'NEW ACTION'
                    # print action
                    if not isinstance(action,int):
                        last_obs, reward, done, info = env.step(action[0])
                        # if np.array_equal(previous_obs,last_obs):
                        #     print 'obs are the same'
                        # else:
                        #     print 'obs not the same'
                        # if np.array_equal(previous_state,temp_state):
                        #     print 'states are the same'
                        # else:
                        #     print 'states are not the same'
                        previous_obs = last_obs
                        # previous_state = temp_state
                    else:
                        last_obs, reward, done, info = env.step(action)
                        # if np.array_equal(previous_obs,last_obs):
                        #     print 'obs are the same'
                        # else:
                        #     print 'obs not the same'
                        # if np.array_equal(previous_state,temp_state):
                        #     print 'states are the same'
                        # else:
                        #     print 'states are not the same'
                        previous_obs = last_obs
                        # previous_state = temp_state
                    # time.sleep(5)
                else:

                    repeat_action_timer += 1
                    action = last_action
                    # print action
                    if not isinstance(action,int):
                        last_obs, reward, done, info = env.step(action[0])
                    else:
                        last_obs, reward, done, info = env.step(action)
                    previous_obs = last_obs
                    # if np.array_equal(previous_state,temp_state):
                    #     print 'states are the same'
                    # else:
                    #     print 'states are not the same'
                    #     previous_state = temp_state
                
            # time.sleep(0.03)
            # rgb = env.render('rgb_array')
            # upscaled=repeat_upsample(rgb,4, 4)
            # viewer.imshow(upscaled)

            if iteration % 25 == 0: # distance check to make sure mario has moved

                curr_distance = info['distance']
                if curr_distance == last_distance:
                    stuck = True # easy way to make mario take another action on the next timestep since he got stuck
                    print 'STUCK CODE RUN'
                last_distance = curr_distance
            replay_buffer.store_effect(obs_idx, action, reward, done)

            # for key in info:
            #     print 'key', key
            #     print 'val for key ', key, 'is', info[key]
            

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


def main():
    # Get Atari games.
    # benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    # task = benchmark.tasks[3]
    task = 'SuperMarioBros-1-1-v0'
    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)
    # print type(env.action_space)
    session = get_session()
    tf.reset_default_graph()

    # smb_learn(env, session, num_timesteps=9999999)
    test_model('SMB_model-241', env)

if __name__ == "__main__":
    main()
