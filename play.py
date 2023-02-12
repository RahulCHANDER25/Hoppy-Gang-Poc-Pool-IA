#!/usr/bin/env python
##
## EPITECH PROJECT, 2023
## rush Pool IA
## File description:
## play file
##

import gym
from AgentClass import *
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import concurrent.futures


my_second_env = gym.make("MsPacman-v0", render_mode='human')

state = my_second_env.reset()
done = False
all_rewards = []
pacman = AgentPacman(my_second_env)
pacman.epsilon = 0
pacman.q_table = torch.load('qtable2')

while not done:
    action = pacman.get_greedy_epsilon_action_best(state)
    new_state, reward, done, infos = my_second_env.step(action)
    all_rewards.append(reward)

my_second_env.close()

fig, ax = plt.subplots()
ax.plot(gaussian_filter1d(all_rewards, sigma=10))
ax.set_title('Rewards while game')
fig.savefig('rewardsgame')
