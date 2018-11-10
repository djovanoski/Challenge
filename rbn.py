import tensorflow as tf
import numpy as np 
import os 
import pandas as pandas


class RestrictedBoltzmanMachinesLayer:
    def __init__(self, num_hidden,name,is_frozen):
        #Name and Number of neurons per layer
        self.num_hidden = num_hidden
        self.name = name
        self.is_frozen = is_frozen
        #W
        #weights and Biases
        self.W = None
        self.bias_visible = None
        self.bias_hidden = np.zeros([1, num_hidden])

    def activate_hidden_from_visible(self, visible):
        net = np.dot(visible, self.W)+ self.bias_hidden
        return self.activate(net)

    def activate_visible_from_hidden(self, hidden):
        net = np.dot(hidden, self.W.T) + self.bias_visible
        return self.activate(net)

    def activate(self, nets):
        return 1.0 / (1.0 + np.exp(-1.0 * nets))

    def sample_hidden_from_visible(self, visible):
        return self.sample(self.activate_hidden_from_visible(visible))

    def sample_visible_from_hidden(self, hidden):
        return self.sample(self.activate_visible_from_hidden(hidden))

    def sample(self, probabilities):
        return np.floor(probabilities + np.random.uniform(size=probabilities.shape))

    def rbn_update(self, x1, learning_rate):
        if self.W is None or self.bias_visible is None:
            self.W = np.random.normal(scale=0.01, size=[x1.shape[1], self.num_hidden])
            self.bias_visible = np.zeros([1, x1.shape[1]])
            #self.bias_visible = np.zeros([x1.shape[0], x1.shape[1]])

        h1 = self.sample_hidden_from_visible(x1)

        x2 = self.sample_visible_from_hidden(h1)
        h_prob = self.activate_hidden_from_visible(x2)
        delta_W = learning_rate * (np.dot(h1.T, x1) - np.dot(h_prob.T, x2))
        self.bias_hidden += learning_rate * (h1 - h_prob)
        self.bias_visible += learning_rate * (x1 - x2)
        self.W += delta_W.T
        
        return np.sum(np.absolute(delta_W))

