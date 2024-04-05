import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sdeint import itoint

from pyPTE.core import pyPTE


def kuramoto(omega_vector, K, N, sigma):
    def f(theta_vector, t):
        #theta_vector = np.atleast_2d(theta_vector)
        theta_vector = np.atleast_2d(theta_vector)
        d_theta_vector = omega_vector + K/N * np.sum(
            np.sin(theta_vector - theta_vector.T), 1)
        p = np.random.normal(0, sigma, N)
        return d_theta_vector + p
    def G(v, t):
        return np.diag(np.ones(N)*sigma)
    return f, G

theta0 = np.array([100.0, 50.0, 0.0])
theta0 = np.arange(0, 10, 1)
print(theta0)
omega = np.array([8, 8, 8])
# omega = np.arange(0, 10, 0.1)
omega = np.ones_like(theta0)*8

f, G = kuramoto(omega, 1, 10, 0.1)

tspan = np.linspace(0, 1, 1000)
solution = itoint(f, G, theta0, tspan)


plt.plot(tspan, solution)
plt.show()

solution = np.mod(solution, 2*np.pi)
solution -= np.pi




plt.plot(tspan, solution)
plt.show()



# phase = np.swapaxes(solution, 0, 1)
phase = solution
delay = pyPTE.get_delay(phase)
phase2 = phase + np.pi
binsize = pyPTE.get_binsize(phase2)
bincount = pyPTE.get_bincount(binsize)
dphase = pyPTE.get_discretized_phase(phase2, binsize)
dPTE, raw_PTE = pyPTE.compute_dPTE_rawPTE(dphase, delay)

print(dPTE)

cmap = sns.diverging_palette(10, 220, sep=80, n=7)
sns.heatmap(dPTE, cmap=cmap, center=0.5)
plt.show()

# for yy in y:
#     plt.plot(x, yy)
# plt.show()

# for yk in yp:
#     plt.plot(x, yk)
# plt.show()
#
# from phase_transfer_entropy.myPTE import get_delay, compute_PTE
# phase = np.swapaxes(yp, 0, 1)
# L, N = phase.shape
# print L, N
# delay = get_delay(phase, L, N)
# print 'delay', delay
# binsize = 3.49*np.mean(np.std(phase,axis=0,ddof=1))*L**(-1.0/3)
# bins_w = np.arange(0, 2*np.pi, binsize)
# Nbins = len(bins_w)
# print compute_PTE(phase, delay, Nbins, L, N)




