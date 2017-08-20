# continue_training_smb.py
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

# from action_space import *
# from control import *


from dqn_utils import *
# from atari_wrappers import *
# from gym.envs.classic_control import rendering

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


OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

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
        graph_name = model_name + '.meta'
        saver = tf.train.import_meta_graph(graph_name)
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        meta_graph = tf.get_default_graph()
        # print graph_name
        print meta_graph

        # for op in meta_graph.get_operations(): # find all saved tensor names
        #     print str(op.name)
        
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
        
        current_qfunc = meta_graph.get_tensor_by_name('current_qfunc/current_q_func_op:0')
        output_layer = meta_graph.get_tensor_by_name('current_qfunc/q_func/action_value/fully_connected_1/BiasAdd:0')

        q_target = meta_graph.get_tensor_by_name('StopGradient_1:0')
        total_error = meta_graph.get_tensor_by_name('Mean:0')

        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "q_func")
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "target_q_func")

        # construct optimization op (with gradient clipping)
        learning_rate = meta_graph.get_tensor_by_name('learning_rate:0')
        optimizer = optimizer_spec.constructor(learning_rate=learning_rate, **optimizer_spec.kwargs)
        # train_fn = minimize_and_clip(optimizer, total_error,
        #              var_list=q_func_vars, clip_val=grad_norm_clipping)

        train_fn = meta_graph.get_tensor_by_name('Adam/Assign_1:0')

        update_target_fn = meta_graph.get_operation_by_name('group_deps')
        # update_target_fn = []
        # for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
        #                            sorted(target_q_func_vars, key=lambda v: v.name)):
        #     update_target_fn.append(var_target.assign(var))
        # update_target_fn = tf.group(*update_target_fn)

        action_predict = tf.argmax(current_qfunc, axis = 1, name = 'action_predict')


        


        ###############
        # RUN ENV     #
        ###############


        model_initialized = True
        num_param_updates = 112
        mean_episode_reward      = -float('nan')
        best_mean_episode_reward = -float('inf')
        last_obs = env.reset()
        LOG_EVERY_N_STEPS = 10000

        repeat_action_timer = 0
        last_action = 0


        tf.global_variables_initializer()

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
        begin_episode = True
        for t in xrange(learning_starts):   # regenerate replay buffer cache

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
                    # print replay_obs.shape
                    # print replay_obs[0].shape
                    # replay_obs = np.reshape(replay_obs, (1,84,84,4))
                    # print type(replay_obs)
                    # print replay_obs[0]
                    # print type(replay_obs)
                    # print replay_obs.shape
                    # assert replay_obs.shape == (1,84,84,4)
                    # act_val = session.run([current_qfunc], feed_dict = {obs_t_ph: [replay_obs]})
                    action = sess.run(action_predict, feed_dict = {obs_t_ph: replay_obs[None, :]})[0]
                    last_action = action
            else:
                repeat_action_timer += 1
                action = last_action


            if not isinstance(action,int):
                last_obs, reward, done, info = env.step(action[0])
            else:
                last_obs, reward, done, info = env.step(action)
            if t % 100 == 0: # distance check to make sure mario has moved

                curr_distance = info['distance']
                if curr_distance == last_distance:
                    begin_episode = True # easy way to make mario take another action on the next timestep since he got stuck
                last_distance = curr_distance

            print 'timestep', t

            # for key in info:
            #     print 'key', key
            #     print 'val for key ', key, 'is', info[key]
            if done:
                print 'Done is True'
                begin_episode = True
                env._close()
                last_obs = env.reset()
                print type(last_obs)

            replay_buffer.store_effect(obs_idx, action, reward, done)


        num_param_updates = 112


        for t in range(1042000,9999999):
            ### 1. Check stopping criterion
            if stopping_criterion is not None and stopping_criterion(env, t):
                break

            ### 2. Step the env and store the transition
            # At this point, "last_obs" contains the latest observation that was
            # recorded from the simulator. Here, your code needs to store this
            # observation and its outcome (reward, next observation, etc.) into
            # the replay buffer while stepping the simulator forward one step.
            # At the end of this block of code, the simulator should have been
            # advanced one step, and the replay buffer should contain one more
            # transition.
            # Specifically, last_obs must point to the new latest observation.
            # Useful functions you'll need to call:
            # obs, reward, done, info = env.step(action)
            # this steps the environment forward one step
            # obs = env.reset()
            # this resets the environment if you reached an episode boundary.
            # Don't forget to call env.reset() to get a new observation if done
            # is true!!
            # Note that you cannot use "last_obs" directly as input
            # into your network, since it needs to be processed to include context
            # from previous frames. You should check out the replay buffer
            # implementation in dqn_utils.py to see what functionality the replay
            # buffer exposes. The replay buffer has a function called
            # encode_recent_observation that will take the latest observation
            # that you pushed into the buffer and compute the corresponding
            # input that should be given to a Q network by appending some
            # previous frames.
            # Don't forget to include epsilon greedy exploration!
            # And remember that the first time you enter this loop, the model
            # may not yet have been initialized (but of course, the first step
            # might as well be random, since you haven't trained your net...)

            #####
            
            # YOUR CODE HERE
            # print '-------------------CODE 2---------------'
            # print last_obs.shape
            # print list(last_obs.shape)
            # print np.array([replay_buffer_size] + list(last_obs.shape)).shape
            # print len([replay_buffer_size] + list(last_obs.shape))
            # test = np.zeros([replay_buffer_size] + list(last_obs.shape))
            # print test
            # print [replay_buffer_size]
            # print list(last_obs.shape)
            # print [replay_buffer_size] + list(last_obs.shape)
            # if not last_obs is None: 

            obs_idx = replay_buffer.store_frame(last_obs)

            if repeat_action_timer >= 8: # repeat the same action for 8 frames (timesteps)

                repeat_action_timer = 0 # reset repeat action timer

                if np.random.random_sample() < exploration.value(t) or not model_initialized or begin_episode:

                    action = env.action_space.sample()
                    if begin_episode:
                        begin_episode = False
                else:
                    replay_obs = replay_buffer.encode_recent_observation()
                    # print replay_obs.shape
                    # print replay_obs[0].shape
                    # replay_obs = np.reshape(replay_obs, (1,84,84,4))
                    # print type(replay_obs)
                    # print replay_obs[0]
                    # print type(replay_obs)
                    # print replay_obs.shape
                    # assert replay_obs.shape == (1,84,84,4)
                    # act_val = session.run([current_qfunc], feed_dict = {obs_t_ph: [replay_obs]})
                    action = sess.run(action_predict, feed_dict = {obs_t_ph: replay_obs[None, :]})[0]
                last_action = action
            else:
                repeat_action_timer += 1
                action = last_action


            if not isinstance(action,int):
                last_obs, reward, done, info = env.step(action[0])
            else:
                last_obs, reward, done, info = env.step(action)

            if t % 100 == 0: # distance check to make sure mario has moved

                curr_distance = info['distance']
                if curr_distance == last_distance:
                    begin_episode = True # easy way to make mario take another action on the next timestep since he got stuck
                last_distance = curr_distance
            print 'timestep', t

            # for key in info:
            #     print 'key', key
            #     print 'val for key ', key, 'is', info[key]
            if done:
                print 'Done is True'
                begin_episode = True
                env._close()
                last_obs = env.reset()
                print type(last_obs)

            replay_buffer.store_effect(obs_idx, action, reward, done)

            #####

            # at this point, the environment should have been advanced one step (and
            # reset if done was true), and last_obs should point to the new latest
            # observation

            ### 3. Perform experience replay and train the network.
            # note that this is only done if the replay buffer contains enough samples
            # for us to learn something useful -- until then, the model will not be
            # initialized and random actions should be taken
            if (t > learning_starts and
                    t % learning_freq == 0 and
                    replay_buffer.can_sample(batch_size)):
                # Here, you should perform training. Training consists of four steps:
                # 3.a: use the replay buffer to sample a batch of transitions (see the
                # replay buffer code for function definition, each batch that you sample
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
                # Remember that you have to update the target network too (see 3.d)!
                # 3.c: train the model. To do this, you'll need to use the train_fn and
                # total_error ops that were created earlier: total_error is what you
                # created to compute the total Bellman error in a batch, and train_fn
                # will actually perform a gradient step and update the network parameters
                # to reduce total_error. When calling session.run on these you'll need to
                # populate the following placeholders:
                # obs_t_ph
                # act_t_ph
                # rew_t_ph
                # obs_tp1_ph
                # done_mask_ph
                # (this is needed for computing total_error)
                # learning_rate -- you can get this from optimizer_spec.lr_schedule.value(t)
                # (this is needed by the optimizer to choose the learning rate)
                # 3.d: periodically update the target network by calling
                # session.run(update_target_fn)
                # you should update every target_update_freq steps, and you may find the
                # variable num_param_updates useful for this (it was initialized to 0)
                #####
                
                # YOUR CODE HERE
                print '-------------------CODE 3---------------', t

                # 3.a
                print '3.a'
                batch_sample = replay_buffer.sample(batch_size)

                # 3.b
                print '3.b'
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
                print '3.c'
                learning_rate_batch = optimizer_spec.lr_schedule.value(t)
                t_error, _ = sess.run([total_error, train_fn], feed_dict = {obs_t_ph: obs_t_batch,
                                                                               act_t_ph: act_t_batch,
                                                                               rew_t_ph: rew_t_batch, 
                                                                               obs_tp1_ph: obs_tp1_batch, 
                                                                               done_mask_ph: done_mask_batch,
                                                                               learning_rate: learning_rate_batch})
                # 3.d
                print '3.d'
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
        replay_buffer_size=125000, #1000000
        batch_size=32,
        gamma=0.99, # 0.99
        learning_starts=30000, # 50000
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10
    )
    env.close()

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
    smb_learn('SMB_model-112', env, num_timesteps = 9999999)

if __name__ == "__main__":
    main()


