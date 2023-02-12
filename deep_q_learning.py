#!/usr/bin/env python3
##
## EPITECH PROJECT, 2023
## rush Poc IA
## File description:
## reinforcement learning
##

EPOCHS = 100
LEARNING_RATE = 0.05
GAMMA = 0.99
STEP_FREQUENCY = 500

import gym
import random
import time
import torch
import copy
from NeuralNetworkClass import *

env = gym.make("MsPacman-v0")
my_dqn = NeuralNetworkDQN(env)
my_second_dqn = NeuralNetworkDQN(env)
state = env.reset()

epsilon = 1.0
steps = 0
loss_fonct = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(my_dqn.parameters(), lr=0.01)

for i in range(EPOCHS):
    state = env.reset()

    done = False
    while not done:
        action = my_dqn.epsilon_greedy(epsilon, state)
        epsilon = max(epsilon * 0.995, 0.1)

        new_state, reward, done, _ = env.step(action=action)

        print(reward)
        action_q_value = my_dqn.forward(torch.Tensor(state))
        y = target_value(reward, GAMMA * (1 - done), my_second_dqn.forward(torch.Tensor(new_state)), action_q_value, LEARNING_RATE)
        print("YES")
        loss = loss_fonct(action_q_value, y)

        print("ok ?")
        # descent_gradient(optimizer=optimizer, loss=loss)
        print("NOW ???")
        state = new_state
        # if steps % STEP_FREQUENCY == 0:
            # my_second_dqn = copy.deepcopy(my_dqn)
        print("DOne ???")
        if done:
            break

