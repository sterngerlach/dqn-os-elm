# coding: utf-8
# dqn-os-elm.py

import numpy as np
import matplotlib.pyplot as plt
import gym
import random

from collections import deque, namedtuple

Transition = namedtuple("Transition",
                        ["state", "action", "next_state", "reward", "done"])

class ExperienceReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.experiences = deque(maxlen=capacity)
    
    def append(self, state, action, next_state, reward, done):
        self.experiences.append(Transition(state, action, next_state, reward, done))
    
    def get_batch(self, batch_size):
        return random.sample(self.experiences, batch_size)
    
    def __len__(self):
        return len(self.experiences)
    
class DqnOsElmAgent(object):
    def __init__(self, num_states, num_actions, num_hidden,
                 batch_size=1, initial_epsilon=0.5, gamma=0.9,
                 experience_replay_capacity=1000):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.initial_epsilon = initial_epsilon
        self.gamma = gamma
        
        self.experience_replay = ExperienceReplayMemory(
            capacity=experience_replay_capacity)
        
        self.alpha = np.random.random((num_states, num_hidden))
        self.b = np.random.random(num_hidden)
        self.P = np.zeros((num_hidden, num_hidden))
        self.beta = np.zeros((num_hidden, num_actions))
        
        self.alpha_old = np.zeros_like(self.alpha)
        self.b_old = np.zeros_like(self.b)
        self.beta_old = np.zeros_like(self.beta)
        
    def initialize(self, experiences):
        # states = np.array([e.state for e in experiences])
        
        # hidden_outputs = self.hidden_output(
        #     states, self.alpha, self.b, self.sigmoid)
        
        # expected_state_values = np.zeros((len(experiences), self.num_actions))
        
        # for i, e in enumerate(experiences):
        #     G = 0.0
        #     t = 0
            
        #     for j in range(i, len(experiences)):
        #         G += np.power(self.gamma, t) * experiences[j].reward
        #         t += 1
            
        #     expected_state_values[i][e.action] = G
        
        # self.P = np.linalg.inv(hidden_outputs.T @ hidden_outputs)
        # self.beta = self.P @ hidden_outputs.T @ expected_state_values
        
        self.P = np.random.random((self.num_hidden, self.num_hidden))
        self.beta = np.random.random((self.num_hidden, self.num_actions))
        
        
        self.update_target()
        
    def update_target(self):
        np.copyto(self.alpha_old, self.alpha)
        np.copyto(self.b_old, self.b)
        np.copyto(self.beta_old, self.beta)
        
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)
        
    def predict(self, state_batch, alpha, b, beta, activation_func):
        hidden_layer_output = activation_func(state_batch.dot(alpha) + b)
        output = hidden_layer_output.dot(beta)
        return output
    
    def hidden_output(self, state_batch, alpha, b, activation_func):
        hidden_layer_output = activation_func(state_batch.dot(alpha) + b)
        return hidden_layer_output
        
    def update(self):
        if len(self.experience_replay) < self.batch_size:
            return
        
        experiences = self.experience_replay.get_batch(self.batch_size)
        
        states = np.array([e.state for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        done_flags = [not e.done for e in experiences]
        non_final_next_states = np.array([e.next_state for e in experiences
                                          if e.done == False])
    
        if len(non_final_next_states) == 0:
            return
        
        hidden_outputs = self.hidden_output(
            states, self.alpha, self.b, self.sigmoid)
        
        next_state_values = np.zeros((self.batch_size, self.num_actions))
        next_state_values[done_flags] = self.predict(
            non_final_next_states, self.alpha_old, self.b_old,
            self.beta_old, self.sigmoid).max(axis=1)
        
        expected_state_values = rewards + self.gamma * next_state_values
        
        self.P -= (self.P @ hidden_outputs.T @ hidden_outputs @ self.P) \
            / (1.0 + hidden_outputs @ self.P @ hidden_outputs.T)
            
        self.beta += self.P @ hidden_outputs.T \
            @ (expected_state_values - hidden_outputs @ self.beta)
            
    def get_action(self, state, episode):
        epsilon = self.initial_epsilon * (1.0 / (episode + 1.0))
        # epsilon = self.initial_epsilon
        
        if epsilon < np.random.uniform():
            state_value = self.predict(np.array([state]), self.alpha, self.b,
                                       self.beta, self.sigmoid)[0]
            return np.argmax(state_value)
        else:
            return np.random.randint(self.num_actions)
    
    def get_greedy_action(self, state):
        state_value = self.predict(np.array([state]), self.alpha, self.b,
                                   self.beta, self.sigmoid)[0]
        return np.argmax(state_value)
        
    def append_experience(self, state, action, next_state, reward, done):
        self.experience_replay.append(state, action, next_state, reward, done)
        
    def play(self, env, episode_num=20):
        for i in range(episode_num):
            state = env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                env.render()
                
                action = self.get_greedy_action(state)
                next_state, reward, done, info = env.step(action)
                
                episode_reward += reward
                state = next_state
            else:
                print("Got reward {}".format(episode_reward))

class DqnOsElmTrainer(object):
    def __init__(self, env_name, gamma=0.9):
        self.env = gym.make(env_name)
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        
        self.agent = DqnOsElmAgent(num_states=self.num_states,
                                   num_actions=self.num_actions,
                                   num_hidden=128,
                                   batch_size=1,
                                   initial_epsilon=0.3,
                                   gamma=gamma)
        
    def __del__(self):
        self.env.close()
        
    def initialize(self):
        experiences = []
        
        state = self.env.reset()
        done = False
        
        # 最初はモンテカルロ法で実行してみる
        while not done:
            action = np.random.randint(self.num_actions)
            next_state, reward, done, info = self.env.step(action)
            
            experiences.append(Transition(state, action, next_state, reward, done))
            
            state = next_state
            
        self.agent.initialize(experiences)
        
    def run(self, episode_num=200):
        recent_rewards = np.zeros(10)
        
        for episode in range(episode_num):
            state = self.env.reset()
            done = False
            step = 0
            
            while not done:
                action = self.agent.get_action(state, episode)
                next_state, reward, done, info = self.env.step(action)
                
                # reward = 0.0 if not done else -1.0 if step < 195 else 1.0
                
                self.agent.append_experience(state, action, next_state, reward, done)
                
                for i in range(10):
                    self.agent.update()
                
                if done:
                    recent_rewards = np.hstack((recent_rewards[1:], step + 1))
                    print("Episode {}, step: {}, average step: {}".format(
                        episode, step + 1, recent_rewards.mean()))
                    
                    if episode % 5 == 0:
                        self.agent.update_target()
                    
                    break
                
                state = next_state
                step += 1
                
    def play(self):
        self.agent.play(self.env)

def main():
    trainer = DqnOsElmTrainer("CartPole-v0", gamma=0.95)
    trainer.initialize()
    trainer.run()
    trainer.play()

if __name__ == "__main__":
    main()
    