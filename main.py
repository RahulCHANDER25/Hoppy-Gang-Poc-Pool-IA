#!/usr/bin/env python
##
## EPITECH PROJECT, 2023
## rush Pool IA
## File description:
## main file
##

EPOCHS = 5000

import gym
# import random
# import time
from AgentClass import *
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import concurrent.futures

env = gym.make("MsPacman-v0")
pacman = AgentPacman(env)

all_rewards = []
all_rewards_another = []
state = []
# pacman.q_table = torch.load('qtable2')

for t in tqdm(range(EPOCHS)):
    # with concurrent.futures.ThreadPoolExecutor(max_workers=500) as executor:
    state = env.reset()
    done = False
    live = 3
    step = 0
    for i in range(20000):
        action = pacman.get_greedy_epsilon_action(state)
        new_state, reward, done, infos = env.step(action)
        if infos['lives'] != live:
            live -= 1
            reward = -5000
            step = 0
        else:
            reward = 1 * (step / 10)
            step += 1
        all_rewards.append(reward)
        # print(reward, step)
        with concurrent.futures.ThreadPoolExecutor(max_workers=500) as executor:
            executor.map(pacman.update_q_table, (new_state, state, reward, action))
        state = new_state
        if done:
            break

    if t % 50 == 0:
        print('Save...')
        torch.save(pacman.q_table, 'qtable2')
        print('DONE !')

        fig, ax = plt.subplots()
        ax.plot(gaussian_filter1d(all_rewards_another, sigma=10))
        ax.set_title('Rewards Mean')
        fig.savefig('rewards')
    all_rewards_another.append(np.mean(all_rewards))
    all_rewards = []
env.close()

torch.save(pacman.q_table, 'qtable2')
# pacman.q_table = 0
# pacman.q_table = torch.load('qtable')

my_second_env = gym.make("MsPacman-v0", render_mode='human')

state = my_second_env.reset()
done = False
all_rewards = []
pacman.epsilon = 0
while not done:
    action = pacman.get_greedy_epsilon_action(state)
    new_state, reward, done, infos = my_second_env.step(action)
    all_rewards.append(reward)

my_second_env.close()

# my_third_env = gym.make("MsPacman-v0", render_mode='human')

# state = my_third_env.reset()
# done = False
# all_rewards = []
# pacman.epsilon = 0
# while not done:
#     action = pacman.get_greedy_epsilon_action(state)
#     new_state, reward, done, infos = my_third_env.step(action)
#     all_rewards.append(reward)

# my_third_env.close()

fig, ax = plt.subplots()
ax.plot(gaussian_filter1d(all_rewards_another, sigma=10))
ax.set_title('Rewards Mean')
fig.savefig('rewards')
