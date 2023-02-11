#!/usr/bin/env python
##
## EPITECH PROJECT, 2023
## rush Pool IA
## File description:
## main file
##

EPOCHS = 100

import gym
import gym_chess
import random
import time
from AgentClass import *

env = gym.make("MsPacman-v0")
pacman = AgentPacman(env)

print(env.observation_space.shape[0])

state = env.reset()

for t in range(EPOCHS):
    state = env.reset()
    all_rewards = 0
    done = False
    while done != True:
        action = pacman.get_greedy_epsilon_action(state)
        new_state, reward, done, infos = env.step(action)
        all_rewards += reward
        pacman.update_q_table(new_state, state, reward, action)
        state = new_state
    print(f"{pacman.epsilon}, {all_rewards}")
