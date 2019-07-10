import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tempfile
import base64
import pprint
import random
import json
import sys
import gym
import io

from gym import wrappers
from collections import deque
from subprocess import check_output
from IPython.display import HTML

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
def action_selection(state, model, episode, n_episodes):
    epsilon =0.995**episode    
 
    values = model.predict(state.reshape(1, 8))[0]
    if np.random.random() < epsilon:
        action = np.random.randint(len(values))
    else:
        action = np.argmax(values)
    return action, epsilon
def neuro_q_learning(env, gamma = 0.99):
    nS = env.observation_space.shape[0]
    nA = env.env.action_space.n
    
    # memory bank
    memory_bank = deque()
    memory_bank_size = 100000
    
    # function approximator
    model = Sequential()
    model.add(Dense(128, input_dim=nS, activation='relu'))
#    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(nA, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=0.00025))

    # constant values
    n_episodes = 1300
    # n_episodes =2
    batch_size = 64
    
    # for statistics
    epsilons = []
    states = []
    actions = []
    accum_reward=0
    rewards=[]
    # interactions
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        
        # each episode
    
        while not done:
            # states.append(state)
            
            # select action
            action, epsilon = action_selection(state, model, episode, n_episodes)
            # epsilons.append(epsilon)
            # actions.append(action)
            
            
            # save history in memory bank
            nstate, reward, done, info = env.step(action)
            accum_reward+=reward
            memory_bank.append((state, action, reward, nstate, done))
            if len(memory_bank) > memory_bank_size:
                memory_bank.popleft()
            
            # iterate to next state
            state = nstate

        # only every few episodes enter training and update neural network weights 
        #if episode % training_frequency == 0 and len(memory_bank) == memory_bank_size:
        
            if  len(memory_bank) > batch_size:
            
            # randomly select batches of samples from the history
            # for training to prevent values spiking due to high 
            # correlation of sequential values
                minibatch = np.array(random.sample(memory_bank, batch_size))

            # extract values by type from the minibatch
                state_batch = np.array(minibatch[:,0].tolist())
                action_batch = np.array(minibatch[:,1].tolist())
                rewards_batch = np.array(minibatch[:,2].tolist())
                state_prime_batch = np.array(minibatch[:,3].tolist())
                is_terminal_batch = np.array(minibatch[:,4].tolist())

            # use the current neural network to predict 
            # current state values and next state values
                state_value_batch = model.predict(state_batch)
                next_state_value_batch = model.predict(state_prime_batch)

            # update the state values given the batch
                for i in range(len(minibatch)):
                    if is_terminal_batch[i]:
                        state_value_batch[i, action_batch[i]] = rewards_batch[i]
                    else:
                        state_value_batch[i, action_batch[i]] = rewards_batch[i] + gamma * np.max(next_state_value_batch[i])
            
            # update the neural network weights
                model.train_on_batch(state_batch, state_value_batch)
        
        epsilons.append(epsilon)
        actions.append(action)
        states.append(state)
        rewards.append(accum_reward)
        if episode%100==0 and episode>1:  
            print('average_reward',accum_reward/100,'epsilon:',epsilon)
            accum_reward=0
    return model, (epsilons, states, actions, rewards)

mdir = tempfile.mkdtemp() 
env = gym.make('LunarLander-v2')
model, stats = neuro_q_learning(env)
epsilons, states, actions, rewards = stats

np.savetxt('epsilons3.dat', epsilons, delimiter=',')
np.savetxt('states3.dat', states, delimiter=',')
np.savetxt('actions3.dat', actions, delimiter=',')
np.savetxt('reward3.dat', rewards, delimiter=',')

test_reward=[]
tot_reward=0
for episode in range(100):
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = np.argmax(model.predict(state.reshape(1, 8))[0])
        nstate, reward, done, info = env.step(action)
        tot_reward+=reward
        state = nstate
    test_reward.append(tot_reward)
np.savetxt('test_rewards3.dat', test_reward, delimiter=',')
