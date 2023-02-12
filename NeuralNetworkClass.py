#!/usr/bin/env python3
##
## EPITECH PROJECT, 2023
## rush Poc IA
## File description:
## Neural Network class
##

import gym
import random
import time
import torch
import torchvision
import pandas
import numpy as np

class NeuralNetworkDQN(torch.nn.Module):

    def __init__(self, space) -> None:
        super().__init__()

        self.q_table = np.zeros([210 * 160, space.action_space.n])
        self.linear1 = torch.nn.Linear(3, 33600)
        self.linear2 = torch.nn.Linear(33600, 16)
        self.linear3 = torch.nn.Linear(16, 1)

        self.actions, self.states, self.rewards = [], [], []
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def epsilon_greedy(self, epsilon, state):
        is_greedy = random.random() > epsilon
        if is_greedy:
            action = np.argmax(self.q_table[state])
        else:
            action = random.randint(0, 8)
        return action

def target_value(r, gamma, q_next, q_current, lr):
    return q_current + lr * (r + gamma * q_next - q_current)

def descent_gradient(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()