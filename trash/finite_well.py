import numpy as np
import matplotlib.pyplot as plt
# from scipy.integrate import quad
from local_util import prettify_plot, basis


def H_nm_finite_well(n, m, N, w, V0, a):
    """
    Analytic Hamiltonian matrix for a finite well potential.
    N: number of basis
    w: frequency
    V0: well depth
    a: well width

    V(x) = 0,  -a/2 < x < a/2
    V(x) = V0,  x > a/2 or x < -a/2
    """
    T = 2*np.pi/w
    if n == m:
        if n <= (N-1)/2:
            kin = 1/2 * (n*w)**2
        else:
            kin = 1/2 * ((n-(N-1)/2) * w)**2

        if n == 0:
            potential = V0 * (T-a)/T
        elif n <= (N-1)/2:
            potential = V0 * (2*n*np.pi - a*n*w + np.sin(2*n*np.pi) 
                              -np.sin(a*n*w))/(2*n*np.pi)
        else:
            nn = n - (N-1)/2
            potential = V0 * (2*nn*np.pi - a*nn*w - np.sin(2*nn*np.pi)
                              +np.sin(a*nn*w))/(2*nn*np.pi)
    else:
        kin = 0
        if (n == 0 and 0 < m <= (N-1)/2) or (m == 0 and 0 < n <= (N-1)/2):
            nn = n+m
            potential = V0 * np.sqrt(2) / (nn*np.pi)
            potential *= (np.sin(nn*np.pi) - np.sin(a*nn*w/2))
        elif (0 < n <= (N-1)/2 and 0 < m <= (N-1)/2):
            potential = V0 * 2 / (m**2*np.pi - n**2*np.pi)
            potential *= (m*np.cos(n*np.pi)*np.sin(m*np.pi) 
                          - n*np.cos(m*np.pi)*np.sin(n*np.pi)
                          - m*np.cos(a*n*w/2)*np.sin(a*m*w/2)
                          + n*np.cos(a*m*w/2)*np.sin(a*n*w/2))
        elif (n > (N-1)/2 and m > (N-1)/2):
            nn = n - (N-1)/2
            mm = m - (N-1)/2
            potential = V0 * 2 / (mm**2*np.pi - nn**2*np.pi)
            potential *= (nn*np.cos(nn*np.pi)*np.sin(mm*np.pi) 
                          - mm*np.cos(mm*np.pi)*np.sin(nn*np.pi)
                          - nn*np.cos(a*nn*w/2)*np.sin(a*mm*w/2)
                          + mm*np.cos(a*mm*w/2)*np.sin(a*nn*w/2))
        else:
            potential = 0
    return kin + potential


def V(x, V0, a):
    return np.sign(np.abs(x) - a/2) * (V0/2) + V0/2


V0 = 50
a = 2
N = 101
w = 1
H = np.zeros((N, N))
H_fin = np.zeros((N, N))
for n in range(N):
    print(n, '\r', end='')
    for m in range(N):
        # H[n, m] = H_nm(n, m, lambda x:V(x, V0, a), N, w)
        H_fin[n, m] = H_nm_finite_well(n, m, N, w, V0, a)

val, vec = np.linalg.eig(H_fin)
idx = np.argwhere(val < V0).T[0]
print('bound states eigenergy is: ', val[idx])
# [ 3.41357099 13.47572274 29.45230759 48.14346422]

prettify_plot()
x = np.linspace(-a, a, 1000)
psi = np.zeros([idx.size, x.size])
for i in range(N):
    for j in range(idx.size):
        psi[j] += np.array([vec[i, idx[j]] * basis(xi, i, N, w) for xi in x])
for j in range(idx.size):
    print('check normalization: ', np.trapz(psi[j]**2, x))
    plt.plot(x, psi[j] + val[idx[j]])

plt.plot(x, V(x, V0, a))
plt.xlabel(r'$x$')
plt.ylabel(r'Eigen wave function $\psi_n(x)$ for $n$th bound state')
plt.show()
