import argparse
import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment import Environment
from noise_class import *
from buffer_class import *
from agent_functions import *

import time

# import tensorflow as tf
import wandb

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], False)
# wandb.login(key = '')
# wandb.init(name='', project="",  entity='')
# config = wandb.config

# set object to collect all parameters and hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)                    # gamma
parser.add_argument('--actor_lr', type=float, default=0.001)                # actor learning rate
parser.add_argument('--critic_lr', type=float, default=0.001)               # critic learning rate
parser.add_argument('--buffer_capacity', type=int, default=10000)           # experience memory buffer size
parser.add_argument('--batch_size', type=int, default=16)                   # minibatch size
parser.add_argument('--tau', type=float, default=0.005)                     # tau (smoothed target update)
parser.add_argument('--state_size', type=int, default=1)                    # image size (input is NxN)
parser.add_argument('--lower_bound', type=float, default=-1)                 # action lower bound
parser.add_argument('--upper_bound', type=float, default=1)                 # action upper bound
parser.add_argument('--std_dev', type=float, default=0.2)                   # exploration noise std
parser.add_argument('--episode_number', type=int, default=10000)            # number of training episodes
parser.add_argument('--episode_steps', type=int, default=150)               # number of steps per episode
parser.add_argument('--show_figure', type=bool, default=False)              # show figure each "save_figure_period" episodes
parser.add_argument('--save_figure', type=bool, default=False)              # save figure each "save_figure_period" episodes
parser.add_argument('--save_figure_period', type=int, default=500)          # how many episodes between saves
parser.add_argument('--model_number', type=int, default=1)                  # number of model (for model saving)
parser.add_argument('--average_window', type=int, default=10)               # n° of episodes on which to compute the reward average
parser.add_argument('--observation_size', type=int, default=15)             # status size (status is NxN)
parser.add_argument('--speed', type=float, default=0.1)                     # drone speed
parser.add_argument('--min_angle', type=float, default= 0)            # min action angle
parser.add_argument('--max_angle', type=float, default= 359.99)                # max action angle
args = parser.parse_args()

# set hyperparameters value
args.batch_size = 16
args.gamma = 0.95
args.tau = 0.005
args.std_dev = 0.3
args.critic_lr = 0.001
args.actor_lr = 0.0001
args.save_figure = True
args.show_figure = False
args.save_figure_period = 1001
args.model_number = '22'
args.episode_number = 1000000
args.episode_steps = 150
args.observation_size = 75
args.speed = 0.12
args.average_window = 100

# save hyperparameters in wandb
# config.gamma = args.gamma
# config.buffer_capacity = args.buffer_capacity
# config.batch_size = args.batch_size
# config.lower_bound = args.lower_bound
# config.upper_bound = args.upper_bound
# config.std_dev = args.std_dev
# config.tau = args.tau
# config.critic_lr = args.critic_lr
# config.actor_lr = args.actor_lr
# config.episode_number = args.episode_number
# config.episode_steps = args.episode_steps
# config.observation_size = args.observation_size
# config.speed = args.speed
# config.average_window = args.average_window
# config.min_action_angle = args.min_angle
# config.max_action_angle = args.max_angle

# create folder and folder paths
map_path = os.path.join('maps', 'training_set_3')
if not os.path.exists('episode_images'):
    os.makedirs('episode_images')
if not os.path.exists('models'):
    os.makedirs('models')
training_path = os.path.join('episode_images', 'training_' + str(args.model_number))
if not os.path.exists(training_path):
    os.makedirs(training_path)
actor_name = os.path.join('models','actor_' + str(args.model_number))
critic_name = os.path.join('models', 'critic_' + str(args.model_number))

# create environment object
env = Environment()
env.set_map_folder(map_path)
env.map_counter = 0
env.set_map_max_number(False, 1)
env.set_step_number(args.episode_steps)
env.set_observation_size(args)
env.set_speed(args.speed)
env.max_vision_angle = 60                           # half vision angle
env.vision_resolution = 150
env.vision_range = 2.5

args.state_size = env.observation_space.shape[0]
args.upper_bound = 1
args.lower_bound = -1
args.num_actions = env.action_space.shape[0]
print("Size of State Space ->  {}".format(args.state_size))
print("Size of Action Space ->  {}".format(args.num_actions))
print("Max Value of Action ->  {}".format(args.max_angle))
print("Min Value of Action ->  {}".format(args.min_angle))

# create noise process
ou_noise = OUActionNoise(np.zeros(1), args.std_dev * np.ones(1))

# create agent nets
actor_model = Actor(args).to(device)
critic_model = Critic(args).to(device)
target_actor = Actor(args).to(device)
target_critic = Critic(args).to(device)

# # initialize weights (initialized with same values for net and target couples)
# target_actor.set_weights(actor_model.get_weights())
# target_critic.set_weights(critic_model.get_weights())

# Initialize target networks
target_actor.load_state_dict(actor_model.state_dict())
target_critic.load_state_dict(critic_model.state_dict())

# Optimizers
actor_optimizer = optim.Adam(actor_model.parameters(), lr=args.actor_lr)
critic_optimizer = optim.Adam(critic_model.parameters(), lr=args.critic_lr)

# initialize buffer
buffer = Buffer(args, device)
ep_reward_list = []                                 # To store reward history of each episode
avg_reward_list = []                                # To store average reward history of last few episodes
reward_data_matrix = np.empty([args.episode_steps, args.episode_number])
reward_data_matrix[:] = np.nan

# TRAINING
for ep in range(1, args.episode_number):
    state = env.reset()
    episode_reward = 0
    step = 0
    time_start_episode = time.time()

    while True:
        time_start = time.time()
        # tf_prev_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        action = policy(actor_model, state_tensor, ou_noise, args)[0]
        next_state, reward, done, info = env.step(action)                        # receive state and reward from environment

        episode_reward += reward
        buffer.record((state, action, reward, next_state))
        buffer.learn(actor_model, critic_model, target_actor, target_critic, actor_optimizer, critic_optimizer, args)
        update_target(target_actor, actor_model, args.tau)
        update_target(target_critic, critic_model, args.tau)
        state = next_state
        reward_data_matrix[step, ep] = reward
        step += 1
        time_end = time.time()
        print("Step {} | Reward = {} | Time per step = {} s".format (step, np.round(reward, 2), np.round(time_end - time_start, 4)))
        if done:                                                                  # end this episode when `done` is True
            break
    # end while

    ep_reward_list.append(episode_reward)
    avg_reward = np.mean(ep_reward_list[-args.average_window:])                  # the average reward is the mean of the last 10 episodes
    avg_reward_list.append(avg_reward)
    time_end_episode = time.time()
    print("Time for episode {}: {} s".format(ep, np.round(time_end_episode - time_start_episode, 2)))
    print("Episode {} | Episode Reward = {} | Average Reward = {} | Episode steps = {} | Total steps = {} | Map N° {}".format(ep, np.round(episode_reward, 2), np.round(avg_reward, 2), env.step_number, env.total_steps, env.map_counter))
    # wandb.log({'Reward': episode_reward, 'Average Reward': avg_reward, 'Episode Steps': env.step_number})
    if ep % args.save_figure_period == 0:
        figure_name = os.path.join('episode_images', 'training_' + str(args.model_number), 'episode_' + str(ep))
        env.visualize(ep, args.show_figure, args.save_figure, figure_name, episode_reward)
        actor_model.save(actor_name + '_ep_' + str(ep))
        critic_model.save(critic_name + '_ep_' + str(ep))
        np.savetxt('training_' + str(args.model_number) + '.csv', reward_data_matrix, fmt='%0.4f', delimiter=",")

# save models
np.savetxt('training_' + str(args.model_number) + '.csv', reward_data_matrix, fmt='%0.4f', delimiter=",")
actor_model.save(actor_name + '_final')
critic_model.save(critic_name + '_final')
# Save final models
torch.save(actor_model.state_dict(), actor_name.replace('.pt', '_final.pt'))
torch.save(critic_model.state_dict(), critic_name.replace('.pt', '_final.pt'))
print(actor_model.summary(0))
print(critic_model.summary(0))
