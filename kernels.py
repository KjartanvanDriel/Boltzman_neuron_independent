import numpy as np

def exponential_kernel(t, dt, tau):
    return np.exp(-t*dt/tau)

def exponential_recurrent_kernel(t, dt, tau):
    return tau/2*np.exp(-t*dt/tau)

def exponential_kernel_K(tau):
    return tau



def double_exponential_kernel(t, dt, alpha, beta, C, S):
    return C*alpha*beta/(beta - alpha) * ( np.exp(-alpha * dt * t/ S)- np.exp(-beta* dt*t/S) )

def double_exponential_recurrent_kernel(t,dt, alpha, beta,C, S):
    return C*C*S*alpha*beta/(2*( beta - alpha)*(alpha + beta)) *(beta * np.exp(-alpha*t*dt/S) - alpha * np.exp(-beta* t*dt/S))

def double_exponential_K(alpha, beta, C, S):
    return C*S

#print(double_exponential_kernel(np.arange(1000), 0.2, 2, 4, 1, 100))


