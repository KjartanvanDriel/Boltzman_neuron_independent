import numpy as np

import matplotlib.pyplot as plt
import kernels
plt.style.use("./dark_theme.mplstyle")


class Neurons:

    def __init__(self, size, threshhold, d) -> None:

        self.dt = 0.2  #ms
        self.tau = 100 #ms

        self.alph = np.exp(- self.dt/self.tau)
        self.alph2 = np.exp(- self.dt/self.tau)

        self.kernel = kernels.double_exponential_kernel(np.arange(2000), self.dt, 2, 4, 1, 100)
        self.kernel2 = self.kernel**2

        self.d = d

        self.x = 1

        self.size = size
        self.T = np.ones(size)*threshhold + np.random.randn(size)*0.1
        self.p = np.ones(size)*0
        self.s = np.ones(size)*0
        self.r = np.ones(size)*0
        self.R = np.ones(size)*0
        self.C = np.ones(size)*0
        self.U = 0
        self.G = np.ones(size)*0

        self.neuron_log = np.array([ [self.s, self.p, self.p, self.r, self.R, self.C, self.T, self.G] ])
        self.global_log = np.array([ [self.U]])

        self.plot_log = np.array([ [self.s, self.p, self.p, self.r, self.R, self.C, self.T, self.G] ])
    
    def update(self, t) -> None:

        self.p = 1/(1 + np.exp(-self.T))

        prob = np.random.rand(self.size)

        self.s = 1*(self.p > prob)


        self.r = np.tensordot(self.neuron_log[ :-len(self.kernel) - 1:-1, 0] , self.kernel[:len(self.neuron_log)], axes=(0,0))
        self.U = self.x - self.d * np.sum(self.r)

        if t % 500 == 0:
            self.plot_log = np.append(self.plot_log, [[self.s, self.p, self.p*(1-self.p), self.r, self.R, self.C, self.T, self.G]], axis=0)

        if len(self.global_log) < 3*len(self.kernel):
            self.neuron_log = np.append(self.neuron_log, [[self.s, self.p, self.p*(1-self.p), self.r, self.R, self.C, self.T, self.G]], axis=0)
            self.global_log = np.append(self.global_log, [[self.U]], axis=0)
        else:
            self.neuron_log = np.append(self.neuron_log[1:], [[self.s, self.p, self.p*(1-self.p), self.r, self.R, self.C, self.T, self.G]], axis=0)
            self.global_log = np.append(self.global_log[1:], [[self.U]], axis=0)


        self.R = 2*self.d*np.tensordot(self.neuron_log[ :-len(self.kernel) - 1:-1, 2] , self.kernel[:len(self.neuron_log)], axes=(0,0))
        self.C = self.d**2*np.tensordot(self.neuron_log[ :-len(self.kernel) - 1:-1, 2] * (-1) ** (self.neuron_log[ :-len(self.kernel) - 1:-1, 0]) , self.kernel2[:len(self.neuron_log)], axes=(0,0))
        self.G = self.R * self.U + self.C
        eta = 0.0001
        self.T = self.T + eta * self.G 


        self.neuron_log[-1] = np.array([self.s, self.p, self.p*(1-self.p), self.r, self.R, self.C, self.T, self.G])



    def clean(self) -> None:
        self.neuron_log = np.array(self.neuron_log) 
        self.neuron_log = np.einsum('ijk->kji', self.neuron_log)

        self.plot_log = np.array(self.plot_log) 
        self.plot_log = np.einsum('ijk->kji', self.plot_log)

        self.global_log = np.array(self.global_log) 
        self.global_log = np.einsum('ij->ji', self.global_log)


    def plot_logs(self):

        tags = ['s', 'p', 'p(1-p)', 'r', 'R', 'C', 'T', 'G']
        for p in range(np.shape(self.neuron_log)[1]):
            plt.figure()
            plt.plot(self.neuron_log[:,p,:].T)
            plt.title(f"{tags[p]} local")
            plt.figure()
            plt.plot(self.plot_log[:,p,:].T)
            plt.title(f"{tags[p]} global")
        
        tags = ['U']
        for p in range(np.shape(self.global_log)[0]):
            plt.figure()
            plt.plot(self.global_log[p])
            plt.title(tags[p])
