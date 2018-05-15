import numpy as np

def kuramoto(omega_vector, K, N, sigma):
    def f(theta_vector, t):
        theta_vector = np.atleast_2d(theta_vector)
        # c = np.sin(theta_vector.T - theta_vector)
        c = np.sin(theta_vector - theta_vector.T)
        Kc = np.multiply(K,c)
        print(K)
        print(Kc)
        d_theta_vector = omega_vector + np.sum(Kc, 1)
        p = np.random.normal(0, sigma, N )
        return d_theta_vector + p
    def G(v, t):
        return np.diag(np.ones(N)*sigma)
    return f, G


N = 20
theta0 = np.array([100.0, 50.0, 0.0])
# theta0 = np.arange(0, N, 1)
theta0 = np.linspace(0,np.pi*2, N)*0

print(theta0)
# omega = np.array([8, 8, 8])
# omega = np.arange(0, 10, 0.1)
omega = np.ones_like(theta0)*1


K = np.zeros((N,N))
print(K)
K[5,8] = 1000
# K[:,3] = 100
f, G = kuramoto(omega, K, N, 0.01)
from sdeint import itoint
tspan = np.linspace(0, 10, 1000)
solution = itoint(f, G, theta0, tspan)
from matplotlib import pyplot as plt
plt.plot(tspan, solution)
plt.show()

solution = np.mod(solution, 2*np.pi)
solution -= np.pi

plt.plot(tspan, solution)
plt.show()




from pyPTE import myPTE
# phase = np.swapaxes(solution, 0, 1)
phase = solution
delay = myPTE.get_delay(phase)
phase2 = phase + np.pi
binsize = myPTE.get_binsize(phase2)
bincount = myPTE.get_bincount(binsize)
dphase = myPTE.get_discretized_phase(phase2, binsize)
dPTE, raw_PTE = myPTE.compute_dPTE_rawPTE(dphase, delay)

print(dPTE)
import seaborn as sns
cmap = sns.diverging_palette(240, 10, as_cmap=True, n=7)
sns.heatmap(dPTE, cmap=cmap, center=0.5, vmin=0.2, vmax=0.8)
plt.show()
sns.heatmap(raw_PTE)
plt.show()

sns.heatmap(K)
plt.show()