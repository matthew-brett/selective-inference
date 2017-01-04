"""
Different randomization options for selective sampler.

Main method used in selective sampler is the gradient method which
should be a gradient of the negative of the log-density. For a 
Gaussian density, this will be a convex function, not a concave function.
"""

from functools import partial

import numpy as np
import regreg.api as rr
from scipy.stats import laplace, norm as ndist

class randomization(rr.smooth_atom):

    def __init__(self, 
                 shape, 
                 density, 
                 grad_negative_log_density, 
                 sampler, 
                 CGF=None, # cumulant generating function and gradient
                 CGF_conjugate=None, # convex conjugate of CGF and gradient
                 lipschitz=1):

        rr.smooth_atom.__init__(self,
                                shape)
        self._density = density
        self._grad_negative_log_density = grad_negative_log_density
        self._sampler = sampler
        self.lipschitz = lipschitz
        
        self.CGF = CGF
        self.CGF_conjugate = CGF_conjugate

    def smooth_objective(self, perturbation, mode='both', check_feasibility=False):
        """
        Compute the negative log-density and its gradient.
        """
        if mode == 'func':
            return self.scale(-np.log(self._density(perturbation)))
        elif mode == 'grad':
            return self.scale(self._grad_negative_log_density(perturbation))
        elif mode == 'both':
            return self.scale(-np.log(self._density(perturbation))), self.scale(self._grad_negative_log_density(perturbation))
        else:
            raise ValueError("mode incorrectly specified")

    def sample(self, size=()):
        return self._sampler(size=size)

    def gradient(self, perturbation):
        """
        Evaluate the gradient of the log-density.

        Parameters
        ----------

        perturbation : np.float

        Returns
        -------

        gradient : np.float
        """
        return self.smooth_objective(perturbation, mode='grad')

    @staticmethod
    def isotropic_gaussian(shape, scale):
        rv = ndist(scale=scale, loc=0.)
        density = lambda x: rv.pdf(x)
        grad_negative_log_density = lambda x: x / scale**2
        sampler = lambda size: rv.rvs(size=shape + size)
        CGF = isotropic_gaussian_CGF(shape, scale)
        CGF_conjugate = isotropic_gaussian_CGF_conjugate(shape, scale)
        return randomization(shape, 
                             density, 
                             grad_negative_log_density, 
                             sampler, 
                             CGF=CGF,
                             CGF_conjugate=CGF_conjugate,
                             lipschitz=1./scale**2)

    @staticmethod
    def gaussian(covariance):
        precision = np.linalg.inv(covariance)
        sqrt_precision = np.linalg.cholesky(precision)
        _det = np.linalg.det(covariance)
        p = covariance.shape[0]
        _const = np.sqrt((2*np.pi)**p * _det)
        density = lambda x: np.exp(-(x * precision.dot(x)).sum() / 2) / _const
        grad_negative_log_density = lambda x: precision.dot(x)
        sampler = lambda size: sqrt_precision.dot(np.random.standard_normal((p,) + size))
        return randomization((p,), 
                             density, 
                             grad_negative_log_density, 
                             sampler, 
                             lipschitz=np.linalg.svd(precision)[1].max())

    @staticmethod
    def laplace(shape, scale):
        rv = laplace(scale=scale, loc=0.)
        density = lambda x: rv.pdf(x)
        grad_negative_log_density = lambda x: np.sign(x) / scale
        sampler = lambda size: rv.rvs(size=shape + size)
        CGF = laplace_CGF(shape, scale)
        CGF_conjugate = laplace_CGF_conjugate(shape, scale)
        return randomization(shape, 
                             density, 
                             grad_negative_log_density, 
                             sampler, 
                             CGF=CGF,
                             CGF_conjugate=CGF_conjugate,
                             lipschitz=1./scale**2)

    @staticmethod
    def logistic(shape, scale):
        # from http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.logistic.html
        density = lambda x: (np.exp(-x / scale) / (1 + np.exp(-x / scale))**2) / scale
        # negative log density is (with \mu=0)
        # x/s + log(s) + 2 \log (1 + e(-x/s))
        grad_negative_log_density = lambda x: (1 - np.exp(-x / scale)) / ((1 + np.exp(-x / scale)) * scale)
        sampler = lambda size: np.random.logistic(loc=0, scale=scale, size=shape + size)
        return randomization(shape, 
                             density, 
                             grad_negative_log_density, 
                             sampler, 
                             lipschitz=.25/scale**2)

# Conjugate generating function for Gaussian

def isotropic_gaussian_CGF(shape, scale): # scale = SD
    return cumulant(shape,
                    lambda x: (x**2).sum() * scale**2 / 2., 
                    lambda x: scale**2 * x)

def isotropic_gaussian_CGF_conjugate(shape, scale):  # scale = SD
    return cumulant_conjugate(shape,
                              lambda x: (x**2).sum() / (2 * scale**2), 
                              lambda x: x / scale**2)

# Conjugate generating function for Laplace

def _standard_laplace_CGF_conjugate(u):
    """
    sup_z uz + log(1 - z**2)
    """
    _zeros = (u == 0)
    root = (-1 + np.sqrt(1 + u**2)) / (u + _zeros)
    value = (root * u + np.log(1 - root**2)).sum()
    return value

def _standard_laplace_CGF_conjugate_grad(u):
    """
    sup_z uz + log(1 - z**2)
    """
    _zeros = (u == 0)
    root = (-1 + np.sqrt(1 + u**2)) / (u + _zeros)
    return root

BIG = 10**10
def laplace_CGF(shape, scale):
    return cumulant(shape,
                    lambda x: -np.log(1 - (scale * x)**2).sum() + BIG * (np.abs(x) > 1),
                    lambda x: 2 * x * scale**2 / (1 - (scale * x)**2))

def laplace_CGF_conjugate(shape, scale):
    return cumulant_conjugate(shape,
                              lambda x: _standard_laplace_CGF_conjugate(x / scale),
                              lambda x: _standard_laplace_CGF_conjugate_grad(x / scale) / scale)

class from_grad_func(rr.smooth_atom):

    """
    take a (func, grad) pair and make a smooth_objective
    """


    def __init__(self,
                 shape,
                 func,
                 grad,
                 coef=1.,
                 offset=None,
                 initial=None,
                 quadratic=None):

        rr.smooth_atom.__init__(self,
                                shape,
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self._func, self._grad = (func, grad)

    def smooth_objective(self, param, mode='both', check_feasibility=False):
        """

        Evaluate the smooth objective, computing its value, gradient or both.

        Parameters
        ----------

        mean_param : ndarray
            The current parameter values.

        mode : str
            One of ['func', 'grad', 'both']. 

        check_feasibility : bool
            If True, return `np.inf` when
            point is not feasible, i.e. when `mean_param` is not
            in the domain.

        Returns
        -------

        If `mode` is 'func' returns just the objective value 
        at `mean_param`, else if `mode` is 'grad' returns the gradient
        else returns both.
        """
        
        param = self.apply_offset(param)

        if mode == 'func':
            return self.scale(self._func(param))
        elif mode == 'grad':
            return self.scale(self._grad(param))
        elif mode == 'both':
            return self.scale(self._func(param)), self.scale(self._grad(param))
        else:
            raise ValueError("mode incorrectly specified")


class cumulant(from_grad_func):
    """
    Class for CGF.
    """
    pass

class cumulant_conjugate(from_grad_func):
    """
    Class for conjugate of a CGF.
    """
    pass


class split(randomization):
    def __init__(self, shape, subsample_size, total_size):

        self.subsample_size = subsample_size
        self.total_size = total_size

        rr.smooth_atom.__init__(self,
                                shape)

    def set_covariance(self, covariance):
        """
        Once covariance has been set, then
        the usual API of randomization will work.
        """
        self._covariance = covariance
        precision = np.linalg.inv(covariance)
        sqrt_precision = np.linalg.cholesky(precision).T
        _det = np.linalg.det(covariance)
        p = covariance.shape[0]
        _const = np.sqrt((2 * np.pi) ** p * _det)
        self._density = lambda x: np.exp(-(x * precision.dot(x)).sum() / 2) / _const
        self._grad_negative_log_density = lambda x: precision.dot(x)
        self._sampler = lambda size: sqrt_precision.dot(np.random.standard_normal((p,) + size))
        self.lipschitz = np.linalg.svd(precision)[1].max()

        def _log_density(x):
            return -np.sum(sqrt_precision.dot(np.atleast_2d(x).T) ** 2, 0) * 0.5 - np.log(_const)

        self._log_density = _log_density

    def smooth_objective(self, perturbation, mode='both', check_feasibility=False):
        if not hasattr(self, "_covariance"):
            raise ValueError('first set the covariance')
        return randomization.smooth_objective(self, perturbation, mode=mode, check_feasibility=check_feasibility)

    def sample(self, size=()):
        if not hasattr(self, "_covariance"):
            raise ValueError('first set the covariance')
        return randomization.sample(self, size=size)

    def gradient(self, perturbation):
        if not hasattr(self, "_covariance"):
            raise ValueError('first set the covariance')
        return randomization.gradient(self, perturbation)

    def randomize(self, loss, epsilon):
        """
        Parameters
        ----------
        loss : rr.glm
            A glm loss with a `subsample` method.
        epsilon : float
            Coefficient in front of quadratic term
        Returns
        -------

        Subsampled loss multiplied by `n / m` where
        m is the subsample size out of a total
        sample size of n.
        The quadratic term is not multiplied by `n / m`
        """
        n, m = self.total_size, self.subsample_size
        inv_frac = n / m
        quadratic = rr.identity_quadratic(epsilon, 0, 0, 0)
        m, n = self.subsample_size, self.total_size  # shorthand
        idx = np.zeros(n, np.bool)
        idx[:m] = 1
        np.random.shuffle(idx)

        randomized_loss = loss.subsample(idx)
        randomized_loss.coef *= inv_frac

        randomized_loss.quadratic = quadratic

        return randomized_loss