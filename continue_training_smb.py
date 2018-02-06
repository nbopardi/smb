# continue_training_smb.py
# This is the script to continue the training process created from run_dqn_smb.py. This script loads model iteration 112 and continues training it until 241 iterations have passed.
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
from collections import namedtuple

from dqn_utils import *

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
    multi_lock = multiprocessing.Lock()
    env._configure(lock = multi_lock)
    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/smb_vid_dir2/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    wrapper = action_space.ToDiscrete()
    env = wrapper(env)
    
    return env



# The optimizer specifications to train the model
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

# Continue training the model from a previous point, assuming the model information is in the same directory
# Parameter - model_name - the model name to continue training from
# Parameter - env - the gym environment to train the model in
# Parameter - optimizer_spec - the optimizer to use to train the model
# Parameter - exploration - the exploration schedule
# Parameter - stopping_criterion - (env, t) -> bool
#                                  Should return true when it's ok for the RL algorithm to stop
#                                  Takes in env and the number of steps executed so far
# Parameter - replay_buffer_size - how many memories to store in the replay buffer
# Parameter - batch_size - how many transitions to sample each time experience is replayed
# Parameter - gamma - discount factor
# Parameter - learning_starts - after how many environment steps to start replaying experiences
# Parameter - learning_freq - how many steps of environment to take between every experience replay
# Parameter - frame_history_len - how many past frames to include as input to the model
# Parameter - target_update_freq - how many experience replay rounds (not steps!) to perform between
#                                  each update to the target Q network
# Parameter - grad_norm_clipping - if not None gradients' norms are clipped to this value
def cont_train_model(model_name,
               env,
               optimizer_spec,
               exploration=LinearSchedule(1000000, 0.1),
               stopping_criterion=None,
               replay_buffer_size = 125000,
               batch_size=32,
               gamma = 0.99,
               learning_starts=30000,
               learning_freq=4,
               frame_history_len = 4,
               target_update_freq=10000,
               grad_norm_clipping=10):

    with tf.Session(graph = tf.Graph()) as sess:

        # Necessary steps to restore graph
        graph_name = model_name + '.meta'
        saver = tf.train.import_meta_graph(graph_name)
        saver.restore(sess, tf.train.latest_checkpoint('./'))
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

        q_target = meta_graph.get_tensor_by_name('StopGradient_1:0')
        total_error = meta_graph.get_tensor_by_name('Mean:0')

        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "q_func")
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "target_q_func")

        # construct optimization op (with gradient clipping)
        learning_rate = meta_graph.get_tensor_by_name('learning_rate:0')
        optimizer = optimizer_spec.constructor(learning_rate=learning_rate, **optimizer_spec.kwargs)

        train_fn = meta_graph.get_tensor_by_name('Adam/Assign_1:0')

        update_target_fn = meta_graph.get_operation_by_name('group_deps')

        # Redefine the action_predict op
        action_predict = tf.argmax(current_qfunc, axis = 1, name = 'action_predict')



        ###############
        # RUN ENV     #
        ###############


        model_initialized = True
        num_param_updates = 112     # the model number that is being trained from
        mean_episode_reward      = -float('nan')
        best_mean_episode_reward = -float('inf')
        LOG_EVERY_N_STEPS = 10000


        tf.global_variables_initializer()   # initialize global variables

        last_obs = env.reset()  # reset the environment
        previous_obs = last_obs # test to see if observations are changing

        replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

        done = False

        iteration = 0

        random_action_counter = 0   # record of random actions done
        repeat_action_timer = 0     # record of how many times an action was repeated
        last_action = 0             # record of last action used
        last_distance = 0           # record of distance covered
        begin_episode = True


        for t in xrange(learning_starts):   # regenerate replay buffer cache since that was lost

            if stopping_criterion is not None and stopping_criterion(env, t):
                break

            obs_idx = replay_buffer.store_frame(last_obs)

            if repeat_action_timer >= 8: # repeat the same action for 8 frames (timesteps)

                repeat_action_timer = 0 # reset repeat action timer

                if begin_episode:

                    action = env.action_space.sample()
                    if begin_episode:
                        begin_episode = False
                else:
                    
                    replay_obs = replay_buffer.encode_recent_observation()
                    action = sess.run(action_predict, feed_dict = {obs_t_ph: replay_obs[None, :]})[0]
                    last_action = action
            else:
                repeat_action_timer += 1
                action = last_action

            # necessary if statement since the action might not be of the correct shape
            if not isinstance(action,int):
                last_obs, reward, done, info = env.step(action[0])
            else:
                last_obs, reward, done, info = env.step(action)
            

            if t % 100 == 0: # distance check to make sure agent has moved

                curr_distance = info['distance']
                if curr_distance == last_distance:
                    begin_episode = True # easy way to make agent take another action on the next timestep since he got stuck
                last_distance = curr_distance

            # print 'timestep', t

            if done:

                begin_episode = True
                env._close()
                last_obs = env.reset()

            replay_buffer.store_effect(obs_idx, action, reward, done)



        # Actual continuation of training begins here
        for t in range(1042000,9999999):
            ### 1. Check stopping criterion
            if stopping_criterion is not None and stopping_criterion(env, t):
                break

            ### 2. Step the env and store the transition
            # At this point, "last_obs" contains the latest observation that was
            # recorded from the simulator.The code needs to store this
            # observation and its outcome (reward, next observation, etc.) into
            # the replay buffer while stepping the simulator forward one step.
            # At the end of this block of code, the simulator should have been
            # advanced one step, and the replay buffer should contain one more
            # transition.
            # Specifically, last_obs must point to the new latest observation.
            # 
            # Note that "last_obs" cannot be used directly as input
            # into the network, since it needs to be processed to include context
            # from previous frames. The replay buffer has a function called
            # encode_recent_observation that will take the latest observation
            # that was pushed into the buffer and compute the corresponding
            # input that should be given to a Q network by appending some
            # previous frames.
            #####
            
            obs_idx = replay_buffer.store_frame(last_obs) # Store most recent observation

            if repeat_action_timer >= 8: # Repeat the same action for 8 frames (timesteps)

                repeat_action_timer = 0 # Reset repeat action timer after 8 frames

                # Take a random action from the environment if within the exploration schedule, the model is not initialized, or episode has not begun
                if np.random.random_sample() < exploration.value(t) or not model_initialized or begin_episode:

                    action = env.action_space.sample()
                
                    if begin_episode:
                        begin_episode = False
                else:
                    replay_obs = replay_buffer.encode_recent_observation()
                    
                    action = sess.run(action_predict, feed_dict = {obs_t_ph: replay_obs[None, :]})[0]
                
                last_action = action
            else:
                repeat_action_timer += 1
                action = last_action

            # necessary if statement since the action might not be of the correct shape
            if not isinstance(action,int):
                last_obs, reward, done, info = env.step(action[0])
            else:
                last_obs, reward, done, info = env.step(action)

            if t % 100 == 0: # distance check to make sure agent has moved

                curr_distance = info['distance']
                if curr_distance == last_distance:
                    begin_episode = True # easy way to make mario take another action on the next timestep since he got stuck
                last_distance = curr_distance
            
            # print 'timestep', t

            if done:

                begin_episode = True
                env._close()
                last_obs = env.reset()

            replay_buffer.store_effect(obs_idx, action, reward, done)

            #####
            # At this point, the environment should have been advanced one step (and
            # reset if done was true), and last_obs should point to the new latest
            # observation

            ### 3. Perform experience replay and train the network.
            # note that this is only done if the replay buffer contains enough samples
            # for us to learn something useful -- until then, the model will not be
            # initialized and random actions should be taken
            if (t > learning_starts and
                    t % learning_freq == 0 and
                    replay_buffer.can_sample(batch_size)):
                # Here, training is performed. Training consists of four steps:
                # 3.a: use the replay buffer to sample a batch of transitions (see the
                # replay buffer code for function definition, each batch that is sampled
                # should consist of current observations, current actions, rewards,
                # next observations, and done indicator).
                # 3.b: initialize the model if it has not been initialized yet; to do
                # that, call
                #    initialize_interdependent_variables(session, tf.global_variables(), {
                #        obs_t_ph: obs_t_batch,
                #        obs_tp1_ph: obs_tp1_batch,
                #    })
                # where obs_t_batch and obs_tp1_batch are the batches of observations at
                # the current and next time step. The boolean variable model_initialized
                # indicates whether or not the model has been initialized.
                # Remember that the target network must be updated too (see 3.d)!
                # 3.c: train the model. To do this, use the train_fn and
                # total_error ops that were created earlier: total_error is what was
                # created to compute the total Bellman error in a batch, and train_fn
                # will actually perform a gradient step and update the network parameters
                # to reduce total_error. When calling session.run on these, the following placeholders
                # need to be populated:
                #       obs_t_ph
                #       act_t_ph
                #       rew_t_ph
                #       obs_tp1_ph
                #       done_mask_ph
                #       (this is needed for computing total_error)
                #       learning_rate --  can be obtained from optimizer_spec.lr_schedule.value(t)
                #       (this is needed by the optimizer to choose the learning rate)
                # 3.d: periodically update the target network by calling
                # session.run(update_target_fn)
                # update every target_update_freq steps, and 
                # variable num_param_updates will be useful for this (it was initialized to 0)
                #####
                
                
                # 3.a

                batch_sample = replay_buffer.sample(batch_size)

                # 3.b

                obs_t_batch = batch_sample[0]
                act_t_batch = batch_sample[1]
                rew_t_batch = batch_sample[2]
                obs_tp1_batch = batch_sample[3]
                done_mask_batch = batch_sample[4]

                if not model_initialized:
                    initialize_interdependent_variables(sess, tf.global_variables(), {
                       obs_t_ph: obs_t_batch,
                       obs_tp1_ph: obs_tp1_batch,
                       rew_t_ph: rew_t_batch
                    })
                    sess.run(update_target_fn)
                    model_initialized = True

                # 3.c

                learning_rate_batch = optimizer_spec.lr_schedule.value(t)
                t_error, _ = sess.run([total_error, train_fn], feed_dict = {obs_t_ph: obs_t_batch,
                                                                               act_t_ph: act_t_batch,
                                                                               rew_t_ph: rew_t_batch, 
                                                                               obs_tp1_ph: obs_tp1_batch, 
                                                                               done_mask_ph: done_mask_batch,
                                                                               learning_rate: learning_rate_batch})
                # 3.d

                if t % target_update_freq == 0:

                    sess.run(update_target_fn)
                    num_param_updates += 1
                    print("Target network parameter update {}".format(num_param_updates))
                    
                    model_name = 'SMB_model'
                    saver.save(sess, model_name, global_step = num_param_updates, write_meta_graph = True)
                    print model_name + ' ' + str(num_param_updates) + ' saved'

                #####

            ### 4. Log progress
            episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
            if len(episode_rewards) > 0:
                mean_episode_reward = np.mean(episode_rewards[-100:])
            if len(episode_rewards) > 100:
                best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
            if t % LOG_EVERY_N_STEPS == 0 and model_initialized:
                print("Timestep %d" % (t,))
                print("mean reward (100 episodes) %f" % mean_episode_reward)
                print("best mean reward %f" % best_mean_episode_reward)
                print("episodes %d" % len(episode_rewards))
                print("exploration %f" % exploration.value(t))
                print("learning_rate %f" % optimizer_spec.lr_schedule.value(t))
                sys.stdout.flush()

# The preprocessing of information before DQN can be run
# Parameter - model_name - the name of the model to load in
# Parameter - env - the gym environment
# Parameter - num_timesteps - the max number of timesteps the algorithm can run for during training
def smb_learn(model_name, env, num_timesteps):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    cont_train_model(
        model_name,
        env,
        optimizer_spec=optimizer,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=125000, 
        batch_size=32,
        gamma=0.99, # 0.99
        learning_starts=30000, 
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10
    )
    env.close()

# Main that gets the environment and the session before training begins
def main():
    
    task = 'SuperMarioBros-1-1-v0'  # The environment to load
    seed = 0                        # Arbitrarily chose a seed of 0
    env = get_env(task, seed)

    session = get_session()
    tf.reset_default_graph()

    # Continue training from model iteration 112
    smb_learn('SMB_model-112', env, num_timesteps = 9999999)

if __name__ == "__main__":
    main()


