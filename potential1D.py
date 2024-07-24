from matplotlib import rc
import numpy as np
from scipy.integrate import quad
from scipy.sparse.linalg import eigs
import functools
import time


def timer(func):
    """Print the runtime of the decorated function.

    ref: https://realpython.com/primer-on-python-decorators/

    Parameters
    ----------
    func : function
           the function to use.
    Returns
    -------
    """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        # print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        print(f"=========== FINISHED in {run_time:.4f} secs. ==============")
        return value
    return wrapper_timer


def prettify_plot():
    """
    change the plot matplotlibrc file

    To use it, please run it before plotting.

    https://matplotlib.org/stable/tutorials/introductory/customizing.html#customizing-with-matplotlibrc-files
    """
    rc('text', usetex=True)
    rc('font', family='serif', serif='Computer Modern Roman', size=8)
    # rc('legend', fontsize=10)
    # rc('mathtext', fontset='cm')
    rc('xtick', direction='in')
    rc('ytick', direction='in')


def basis(x, i, N, w):
    """
    basis.
    x: coordinate
    i: index
    N: number of basis
    w: frequency
    """
    if i == 0:
        return np.sqrt(w/(2*np.pi))
    elif i <= (N-1)/2:
        return np.sqrt(w/np.pi)*np.cos(i*w*x)
    else:
        n = i - (N-1)/2
        return np.sqrt(w/np.pi)*np.sin(n*w*x)


def quad_break(func: callable, a, b, points, **kwargs):
    points = np.sort(points)
    if len(points) == 0:
        return quad(func, a, b, **kwargs)

    if a >= points[0] or b <= points[-1]:
        raise ValueError('points must be inside [a, b]')

    res = [0, 0]
    ps = np.concatenate([[a], points, [b]])
    for i in range(len(ps) - 1):
        resi = np.array(quad(func, ps[i], ps[i+1], **kwargs))
        res[0] += resi[0]
        res[1] += resi[1]
    return res


class Potential1D:
    def __init__(self, V: callable, N, w, points=[], verbose=1):
        self.V = V
        self.N = N
        self.w = w
        self.T = 2*np.pi / w
        self.points = points
        self.verbose = verbose
        self.NN = (N-1)/2  # number of the sin or cos
        self.levels = 0

    def get_eigenvals(self, levels):
        if not hasattr(self, 'eigenvals') or self.levels != levels:
            self._gen_eigenvals(levels)
        return self.eigenvals

    @timer
    def _gen_eigenvals(self, levels):
        self.levels = levels
        self.get_H()
        if self.verbose > 0:
            print('=========== digonal the Hamiltonian ... ===========')
        val, vec = eigs(self.H, k=levels, which='SM')
        self.eigenvals = np.real(val)
        self.vec = np.real(vec)
        if self.verbose > 0:
            print('=========== digonal the Hamiltonian FINISHED! =====')
        return self.eigenvals

    def get_eigenwf(self, xs):
        if not hasattr(self, 'eigenwf'):
            self._gen_eigenwf(xs)
        return self.eigenwf

    @timer
    def _gen_eigenwf(self, xs):
        """get eigen wave function."""
        self.xs = xs
        if not hasattr(self, 'eigenvals'):
            raise ValueError('call get_eigenvals first!')
        # psi = np.zeros([len(self.eigenvals), len(xs)])
        self.eigenwf = []
        if self.verbose > 0:
            print('=========== generate wave function ... ============')
        for veci in self.vec.T:
            psi_i = np.zeros(len(xs))
            for j in range(self.N):
                basis_j_x = np.array([basis(xi, j, self.N, self.w)
                                      for xi in xs])
                psi_i += veci[j]*basis_j_x
            self.eigenwf.append(psi_i)
            if self.verbose > 0:
                print('check normalization: ', np.trapz(psi_i**2, xs))
        if self.verbose > 0:
            print('=========== generate wave function FINISHED! ======')

    def get_H(self):
        if not hasattr(self, 'H'):
            self._gen_H()
        return self.H

    @timer
    def _gen_H(self):
        """
        c: cos term
        s: sin term

        cosA cosB = (   cos(A+B) + cos(A-B)) / 2
        sinA sinB = ( - cos(A+B) + cos(A-B)) / 2
        cosA sinB = (   sin(A+B) - sin(A-B)) / 2
        """
        self._gen_sin_cos_list()
        if self.verbose > 0:
            print('=========== generate Hamiltonian ... ==============')
        self.H = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N)[::-1]:
                if i <= j:
                    self.H[i, j] = self._gen_Hnm(i, j)
                else:
                    self.H[i, j] = self.H[j, i]
        if self.verbose > 0:
            print('=========== generate Hamiltonian FINISHED! ========')

    def _gen_Hnm(self, n, m):
        """
        for n <= m only

        c: cos term
        s: sin term

        cosA cosB = (   cos(A+B) + cos(A-B)) / 2
        sinA sinB = ( - cos(A+B) + cos(A-B)) / 2
        cosA sinB = (   sin(A+B) - sin(A-B)) / 2
        """
        if n == m:
            if n <= self.NN:
                kin = 1/2 * (n*self.w)**2
            else:
                kin = 1/2 * ((n-self.NN) * self.w)**2
        else:
            kin = 0

        fac = 1
        if n == 0:
            fac *= 1/np.sqrt(2)
        if m == 0:
            fac *= 1/np.sqrt(2)

        if (n <= self.NN) and (m <= self.NN):
            nn = n
            mm = m
            int0 = self.cos_list[int(nn+mm)]
            int1 = self.cos_list[int(np.abs(nn-mm))]
            potential = (int0 + int1) / self.T
        elif (n <= self.NN) and (m > self.NN):
            nn = n
            mm = m - self.NN
            int0 = self.sin_list[int(nn+mm)]
            int1 = self.sin_list[int(np.abs(nn-mm))] * np.sign(nn-mm)
            potential = (int0 - int1) / self.T
        elif (n > self.NN) and (m > self.NN):
            nn = n - self.NN
            mm = m - self.NN
            int0 = self.cos_list[int(nn+mm)]
            int1 = self.cos_list[int(np.abs(nn-mm))]
            potential = (- int0 + int1) / self.T
        else:
            raise ValueError('n and m must be <= N/2')

        return fac * potential + kin

    def get_sin_cos_list(self):
        if not hasattr(self, 'sin_list'):
            self._gen_sin_cos_list()
        return self.sin_list, self.cos_list

    @timer
    def _gen_sin_cos_list(self):
        self.sin_list = []
        self.cos_list = []
        for i in range(self.N):
            if self.verbose > 0:
                print('=== calculating cos and sin ===', i, '===', end='\r')
            self.cos_list.append(quad_break(self.V,
                                            -self.T/2, self.T/2,
                                            points=self.points,
                                            weight='cos',
                                            wvar=i*self.w)[0])
            self.sin_list.append(quad_break(self.V, -self.T/2, self.T/2,
                                            points=self.points,
                                            weight='sin',
                                            wvar=i*self.w)[0])
        if self.verbose > 0:
            print('=========== calculating cos and sin  FINISHED! ====')
