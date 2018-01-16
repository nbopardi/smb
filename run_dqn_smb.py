# run_dqn_smb.py
# This is the script that was used to initially set up the training process for the model up until iteration 112. Training was continued with some slight modifications
# (see continue_training_smb.py for the latest training script)
import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import multiprocessing
import dqn
import action_space
import control

from dqn_utils import *

# The model for the q function. The layers include 3 convolutional layers of 32, 64, and 64 ReLU neurons, and two fully connected layers of 512 and 10 ReLU neurons.
# Parameter - img_in - the raw image that is fed into the net
# Parameter - num_actions - the number of possible actions for the agent
# Parameter - reuse - boolean to determine if the weights should be reused in the same scope 
def smb_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out
# The preprocessing of information before DQN can be run
# Parameter - env - the gym environment
# Parameter - session - the current tensorflow session
# Parameter - num_timesteps - the max number of timesteps the algorithm can run for during training
def smb_learn(env,
                session,
                num_timesteps):
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

    dqn.learn(
        env,
        q_func=smb_model,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=150000, #1000000
        batch_size=32,
        gamma=0.99, # 0.99
        learning_starts=30000, # 50000
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10
    )
    env.close()

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
    # print type(env.action_space)
    
    return env

# Main that gets the environment and the session before training begins
def main():

    task = 'SuperMarioBros-1-1-v0'  # The environment to load
    seed = 0                        # Arbitrarily chose a seed of 0
    env = get_env(task, seed)
    session = get_session()

    smb_learn(env, session, num_timesteps=9999999)

if __name__ == "__main__":
    main()
