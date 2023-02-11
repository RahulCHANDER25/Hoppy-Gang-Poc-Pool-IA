#!/usr/bin/env python3
##
## EPITECH PROJECT, 2023
## rush Poc IA
## File description:
## Class Neural Network
##

import torch
import torchvision
import pandas
import numpy as np
import matplotlib as plt
import random

GAMMA = 0.99
LR = 0.5

class AgentPacman():

    def __init__(self, space) -> None:
        self.epsilon = 1.0
        self.q_table = np.zeros([210*160, space.action_space.n])

    def get_greedy_epsilon_action(self, state):
        self.epsilon = max(self.epsilon * 0.995, 0.2)
        is_a_greedy_action = random.random() > self.epsilon
        if is_a_greedy_action:
            action = np.argmax(self.q_table[state])
        else:
            action = random.randint(0, 8)
        return action

    def update_q_table(self, new_state, state, reward, action):
        estimate_value = np.max(self.q_table[new_state[:,:,0]])
        new_value = reward + GAMMA * estimate_value
        temp_diff = new_value - self.q_table[state[:,:,0]][action]
        self.q_table[state[:,:,0]][action] = self.q_table[state[:,:,0]][action] + LR * temp_diff
