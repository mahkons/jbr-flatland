import torch
import numpy as np

from queue import PriorityQueue

# simple bandit
class ThresholdBandit():
    def __init__(self):
        self.N = 100
        self.points = np.arange(self.N) / self.N
        self.active = np.zeros(self.N, dtype=np.bool)
        self.timer = 1
        
        self.cnt_pulls = np.zeros(self.N) + 1e-15
        self.sum_pulls = np.zeros(self.N)

        self.delayed_mean = np.zeros(self.N)
        self.gamma = 0.99

    def step(self, threshold, reward):
        self.timer += 1
        pos = int(threshold * self.N + 0.5)
        self.cnt_pulls[pos] += 1
        self.sum_pulls[pos] += reward
        self.delayed_mean[pos] = self.gamma * self.delayed_mean[pos] + (1 - self.gamma) * reward

        self.exploration_factor = np.sqrt(2 * np.log(self.timer) / (self.cnt_pulls + 1))
        self.value = self.delayed_mean + self.exploration_factor

    def choose_threshold(self):
        threshold_pos = np.argmax(self.value)
        return threshold_pos / self.N
