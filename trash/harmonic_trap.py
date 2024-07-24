import numpy as np
import matplotlib.pyplot as plt
from local_util import prettify_plot, Potential1D


w0 = 1
N = 2001
w = .1
levels = 5


def V(x, w0=w0):
    return (w0*x)**2 / 2


p1d = Potential1D(V=V, N=N, w=w, verbose=1, points=[])
vals = p1d.get_eigenvals(levels=levels)
wf = p1d.get_eigenwf(xs=np.linspace(-10, 10, 1000))
print('bound states eigenergy is: ', vals)


prettify_plot()

for j in range(levels):
    plt.plot(p1d.xs, wf[j] + vals[j],
             label=r'$E_n='+f'{vals[j]:.2f}'+r'$')

plt.plot(p1d.xs, V(p1d.xs))
plt.xlabel(r'$x$')
plt.ylabel(r'Eigen wave function $\psi_n(x)$ for $n$th bound state')
plt.xlim(-5, 5)
plt.ylim(-.5, 6.5)
plt.legend()
plt.show()
