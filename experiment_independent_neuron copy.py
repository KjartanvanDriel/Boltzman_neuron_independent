import sys

import numpy as np
import matplotlib.pyplot as plt

import model
import sb
import prepare_input

import evaluate

import plot_func

#Input parameters:
index = int(sys.argv[1])

parameter_file = open("parameters",'r')
parameter_file_lines = parameter_file.readlines()
parameter_file_keys = parameter_file_lines[0].replace("\n","").split("|")
parameter_file_parameters = parameter_file_lines[index].replace("\n","").split("|")

parameters = { k: float(v) for (k,v) in zip(parameter_file_keys,parameter_file_parameters) }

timescale = parameters["timescale"]
decoder = parameters["decoder"]
deltaU = parameters["deltaU"]

print(parameters)

#Input functions
inp_func = lambda t, dt : [1]

#Snapshot input functions
G = sb.Group(1, 'G', spike_rate = 30, deltaU=deltaU, T_learning_rate=3e-3)
G.kernel_params = [timescale]
G.T = np.ones(np.shape(G.T))*decoder**2*timescale/4
I = sb.Input(1, inp_func, 'I', inp_func)

I_G = sb.fw_Connection(I, G, 'I -> G', learning_rate = 3e-3)
I_G.weights = np.array([[decoder]])

#We desire no recurrent connections yet
#G_re = sb.re_Connection(G, G, 'G re G')

sb.init(experiment_name = f"independent_neuron_{index}", T = 500000, dt = 0.2, learn = True, snapshots=2, snapshot_timesteps=5000)

sb.run()

