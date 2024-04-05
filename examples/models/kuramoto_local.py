import numpy as np

def kuramoto(omega_vector, K, N, sigma):
    def f(theta_vector, t):
        theta_vector = np.atleast_2d(theta_vector)
        # c = np.sin(theta_vector.T - theta_vector)
        c = np.sin(theta_vector - theta_vector.T)
        Kc = np.multiply(K,c)
        # print(K)
        # print(Kc)
        d_theta_vector = omega_vector + np.sum(Kc, 1)
        p = np.random.normal(0, sigma, N )
        print(p)
        return d_theta_vector + p
    def G(v, t):
        return np.diag(np.ones(N)*sigma)
    return f, G


N = 10
# theta0 = np.array([100.0, 50.0, 0.0])
# theta0 = np.arange(0, N, 1)
theta0 = np.linspace(0,np.pi*2, N)

print(theta0)
# omega = np.array([8, 8, 8])
# omega = np.arange(0, 10, 0.1)
omega = np.ones_like(theta0)*5
# omega = np.linspace(0, np.pi*2, N)

K = np.zeros((N,N))
print(K)
K[:,:] = 0
K[2,3] = 0

dt = 0.01
t_end = 1
steps = t_end / dt
# K[:,3] = 100
f, G = kuramoto(omega, K, N, 0.01)
from sdeint import itoint
tspan = np.linspace(0, t_end, steps)
solution = itoint(f, G, theta0, tspan)
from matplotlib import pyplot as plt
plt.figure(1)
plt.subplot(211)
plt.plot(tspan, solution)

solution = np.mod(solution, 2*np.pi)
solution -= np.pi
plt.subplot(212)
plt.plot(tspan, solution)
plt.show()

import itertools

dt = 0.1
t_end = 10
steps = t_end / dt
from sdeint import itoint
def repetitive_sims(K, N, theta0, omega, Nsims):
    solutions = list()
    for _ in itertools.repeat(None, Nsims):
        f, G = kuramoto(omega, K, N, 1)
        tspan = np.linspace(0, t_end, steps)
        solution = itoint(f, G, theta0, tspan)
        solutions.append(solution)
        solution = np.mod(solution, 2 * np.pi)
        solution -= np.pi
    return solutions

# Nsim = 10
# solutions = repetitive_sims(K, N, theta0, omega, Nsim)

from matplotlib import pyplot as plt
# for solution in solutions:
#     plt.plot(solution)
#     plt.show()
# dPTEs = list()
# for solution in solutions:
#     from pyPTE import myPTE
#     # phase = np.swapaxes(solution, 0, 1)
#     phase = solution
#     delay = myPTE.get_delay(phase)
#     phase2 = phase + np.pi
#     binsize = myPTE.get_binsize(phase2)
#     bincount = myPTE.get_bincount(binsize)
#     dphase = myPTE.get_discretized_phase(phase2, binsize)
#     dPTE, raw_PTE = myPTE.compute_dPTE_rawPTE(dphase, delay)
#     dPTEs.append(dPTE)
#
# average_dPTE = np.sum(dPTEs) / len(dPTEs)

from pyPTE.core import pyPTE
# phase = np.swapaxes(solution, 0, 1)
phase = solution
delay = pyPTE.get_delay(phase)
print('delay', delay, delay*dt)
phase2 = phase + np.pi
binsize = pyPTE.get_binsize(phase2)
bincount = pyPTE.get_bincount(binsize)
dphase = pyPTE.get_discretized_phase(phase2, binsize)
dPTE, raw_PTE = pyPTE.compute_dPTE_rawPTE(dphase, delay)



import seaborn as sns
plt.figure(1)

cmap = sns.diverging_palette(240, 10, as_cmap=True, n=7)
plt.subplot(131)
plt.axis('equal')
sns.heatmap(dPTE, cmap=cmap, center=0.5, vmin=0.2, vmax=0.8)
plt.subplot(132)
plt.axis('equal')
sns.heatmap(raw_PTE)
plt.subplot(133)
plt.axis('equal')
sns.heatmap(K)
plt.show()

print(raw_PTE)
