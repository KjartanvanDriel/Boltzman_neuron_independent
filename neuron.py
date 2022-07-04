import numpy as np

import matplotlib.pyplot as plt


class Neurons:

    def __init__(self, size, threshhold, d) -> None:

        self.dt = 0.2  #ms
        self.tau = 100 #ms

        self.alph = np.exp(- self.dt/self.tau)
        self.alph2 = np.exp(- self.dt/self.tau)

        self.d = d

        self.x = 1

        self.size = size
        self.T = np.ones(size)*threshhold
        self.neuron_log = []

        self.global_log = []

        self.s = np.ones(size)*0
        self.r = np.ones(size)*0
        self.R = np.ones(size)*0
        self.C = np.ones(size)*0
        self.U = np.ones(size)*0
    
    def update(self, p = None) -> None:

        p_spike = 1/(1 + np.exp(-self.T))
        if p == None:
            p = np.random.rand(self.size)

        self.s = 1*(p_spike > p)

        self.r = self.r * self.alph + self.s

        self.R = self.R * self.alph + 2*self.d * p_spike*(1-p_spike) 
        self.C = self.C * self.alph2 + self.d**2 * p_spike*(1-p_spike) * (-1)**self.s

        self.U = self.x - self.d * np.sum(self.r)

        self.G = self.R * self.U# + self.C

        eta = 0.0001
        self.T = self.T + eta * self.G 

        self.neuron_log.append([self.s, self.r, self.R, self.C, self.T, self.G])
        self.global_log.append([self.U])

    def clean(self) -> None:
        self.neuron_log = np.array(self.neuron_log) 
        self.neuron_log = np.einsum('ijk->kji', self.neuron_log)
        self.global_log = np.array(self.global_log) 
        self.global_log = np.einsum('ij->ji', self.global_log)


    def plot_logs(self):
        tags = ['s', 'r', 'R', 'C', 'T', 'G']
        for p in range(np.shape(self.neuron_log)[1]):
            plt.figure()
            plt.plot(self.neuron_log[:,p,:].T)
            plt.title(tags[p])
        
        tags = ['U']
        for p in range(np.shape(self.global_log)[0]):
            plt.figure()
            plt.plot(self.global_log[p])
            plt.title(tags[p])
