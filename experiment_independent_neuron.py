import sys

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import neuron_double_exponent
import neuron




ball = neuron_double_exponent.Neurons(100,-5,0.5)


for t in tqdm(range(15000)):
    ball.update(t)

ball.clean()


ball.plot_logs()

plt.show()
