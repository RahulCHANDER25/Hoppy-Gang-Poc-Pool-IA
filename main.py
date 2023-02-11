#!/usr/bin/env python
##
## EPITECH PROJECT, 2023
## rush Pool IA
## File description:
## main file
##

EPOCHS = 1000

import gym
import random
import time
from AgentClass import *

env = gym.make("MsPacman-v0")
pacman = AgentPacman(env)

# print(gym.RewardWrapper(env).reward(1.0))
# print(env.observation_space.shape[0])

state = env.reset()

all_rewards = []
# print(pacman.q_table)
for t in range(EPOCHS):
    state = env.reset()
    done = False
    live = 3
    while done != True:
        action = pacman.get_greedy_epsilon_action(state)
        new_state, reward, done, infos = env.step(action)
        if infos['lives'] != live:
            live -= 1
            reward = -50
        if reward == 0:
            reward = -1
        all_rewards.append(reward)
        pacman.update_q_table(new_state, state, reward, action)
        state = new_state
    if t % 10 == 0:
        print(np.mean(all_rewards), pacman.epsilon)
        all_rewards = []

env.close()
my_second_env = gym.make("MsPacman-v0", render_mode='human')

state = my_second_env.reset()
last_game = False
all_rewards = []
pacman.epsilon = 0
while not last_game:
    action = pacman.get_greedy_epsilon_action(state)
    new_state, reward, done, infos = my_second_env.step(action)
    all_rewards.append(reward)
print(f"{pacman.epsilon}, {all_rewards}")
my_second_env.close()
