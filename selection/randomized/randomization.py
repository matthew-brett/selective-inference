"""
Different randomization options for selective sampler.

Main method used in selective sampler is the gradient method which
should be a gradient of the negative of the log-density. For a 
Gaussian density, this will be a convex function, not a concave function.
"""
from __future__ import division

import numpy as np
import regreg.api as rr
from scipy.stats import laplace, norm as ndist

class randomization(rr.smooth_atom):

    def __init__(self, shape, density, grad_negative_log_density, sampler, lipschitz=1):
        rr.smooth_atom.__init__(self,
                                shape)
        self._density = density
        self._grad_negative_log_density = grad_negative_log_density
        self._sampler = sampler
        self.lipschitz = lipschitz

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

    def randomize(self, loss, epsilon=0):
        """
        Randomize the loss.
        """

        randomized_loss = rr.smooth_sum([loss])
        _randomZ = self.sample()
        randomized_loss.quadratic = rr.identity_quadratic(epsilon, 0, -_randomZ, 0)
        return randomized_loss

    @staticmethod
    def isotropic_gaussian(shape, scale):
        """
        Isotropic Gaussian with SD `scale`.

        Parameters
        ----------

        shape : tuple
            Shape of noise.

        scale : float
            SD of noise.
        """
        rv = ndist(scale=scale, loc=0.)
        density = lambda x: rv.pdf(x)
        grad_negative_log_density = lambda x: x / scale**2
        sampler = lambda size: rv.rvs(size=shape + size)
        return randomization(shape, density, grad_negative_log_density, sampler, lipschitz=1./scale**2)

    @staticmethod
    def gaussian(covariance):
        """
        Gaussian noise with a given covariance.

        Parameters
        ----------

        covariance : np.float((*,*))
            Positive definite covariance matrix. Non-negative definite
            will raise an error.
        """
        precision = np.linalg.inv(covariance)
        sqrt_precision = np.linalg.cholesky(precision)
        _det = np.linalg.det(covariance)
        p = covariance.shape[0]
        _const = np.sqrt((2*np.pi)**p * _det)
        density = lambda x: np.exp(-(x * precision.dot(x)).sum() / 2) / _const
        grad_negative_log_density = lambda x: precision.dot(x)
        sampler = lambda size: sqrt_precision.dot(np.random.standard_normal((p,) + size))
        return randomization((p,), density, grad_negative_log_density, sampler, lipschitz=np.linalg.svd(precision)[1].max())

    @staticmethod
    def laplace(shape, scale):
        """
        Standard Laplace noise multiplied by `scale`

        Parameters
        ----------

        shape : tuple
            Shape of noise.

        scale : float
            Scale of noise.
        """
        rv = laplace(scale=scale, loc=0.)
        density = lambda x: rv.pdf(x)
        grad_negative_log_density = lambda x: np.sign(x) / scale
        sampler = lambda size: rv.rvs(size=shape + size)
        return randomization(shape, density, grad_negative_log_density, sampler, lipschitz=1./scale**2)

    @staticmethod
    def logistic(shape, scale):
        """
        Standard logistic noise multiplied by `scale`

        Parameters
        ----------

        shape : tuple
            Shape of noise.

        scale : float
            Scale of noise.
        """
        # from http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.logistic.html
        density = lambda x: (np.exp(-x / scale) / (1 + np.exp(-x / scale))**2) / scale
        # negative log density is (with \mu=0)
        # x/s + log(s) + 2 \log (1 + e(-x/s))
        grad_negative_log_density = lambda x: (1 - np.exp(-x / scale)) / ((1 + np.exp(-x / scale)) * scale)
        sampler = lambda size: np.random.logistic(loc=0, scale=scale, size=shape + size)
        return randomization(shape, density, grad_negative_log_density, sampler, lipschitz=.25/scale**2)


class split(randomization):

    def __init__(self, loss, active, solve_args={'min_its':50, 'tol':1.e-10}):

        from .M_estimator import restricted_Mest

        self.n, self.p = loss.X.shape
        self.fraction = loss.fraction
        self.subsample = loss.subsample
        self.X, self.y = loss.X, loss.y
        self.X1, self.y1  = self.X[self.subsample,:], self.y[self.subsample]

        subsample_set = set(self.subsample)
        total_set = set(np.arange(self.n))
        outside_subsample = np.asarray([item for item in total_set if item not in subsample_set])
        self.y2 = self.y[outside_subsample]

        #full_glm_loss = loss.full_loss
        #beta_overall = restricted_Mest(full_glm_loss, np.ones(self.p, dtype=bool), solve_args=solve_args)
        sub_glm_loss1 = loss.sub_loss
        beta_overall1 = np.zeros(self.p)
        beta_overall1[active] = restricted_Mest(sub_glm_loss1, active, solve_args=solve_args)

        beta_overall2 = np.zeros(self.p)
        sub_glm_loss2 = rr.glm.logistic(self.X[outside_subsample], self.y[outside_subsample])
        beta_overall2[active] = restricted_Mest(sub_glm_loss2, active, solve_args=solve_args)

        def _boot_covariance(indices):
            X_star, y_star = self.X[indices], self.y[indices]
            X1_star, y1_star = X_star[self.subsample], y_star[self.subsample]
            X2_star, y2_star = X_star[outside_subsample], y_star[outside_subsample]

            _boot_mu1 = lambda X1: sub_glm_loss1.saturated_loss.smooth_objective(X1.dot(beta_overall1), 'grad') + self.y1

            _boot_mu2 = lambda X2: sub_glm_loss2.saturated_loss.smooth_objective(X2.dot(beta_overall2), 'grad') + self.y2

            #subsample = np.random.choice(n, size=(m,), replace=False)
            score1 = X1_star.T.dot(y1_star - _boot_mu1(X1_star))
            score2 = X2_star.T.dot(y2_star - _boot_mu2(X2_star))
            result = (1-self.fraction)*score1 - (self.fraction*score2)
            #result = np.dot(X_star[self.subsample].T, y_star[self.subsample]) - (self.fraction*np.dot(X_star.T, y_star))
            return result

        def _nonparametric_covariance_estimate(nboot=10000):
            results = []
            for i in range(nboot):
                indices = np.random.choice(self.n, size=(self.n,), replace=True)
                results.append(_boot_covariance(indices))

            mean_results = np.zeros(self.p)
            for i in range(nboot):
                mean_results = np.add(mean_results, results[i])

            mean_results /= nboot

            covariance = np.zeros((self.p,self.p))
            for i in range(nboot):
                covariance = np.add(covariance, np.outer(results[i]-mean_results, results[i]-mean_results))

            return covariance/float(nboot)

        self.covariance = _nonparametric_covariance_estimate()
        print np.diag(self.covariance)
        #covariance_inv = np.linalg.inv(self.covariance)

        #from scipy.stats import multivariate_normal

        #density = lambda x: multivariate_normal.pdf(x, mean=np.zeros(p), cov=self.covariance)
        #grad_negative_log_density = lambda x: covariance_inv.dot(x)
        #sampler = lambda size: np.random.multivariate_normal(mean=np.zeros(p), cov=self.covariance, size=size)

        def gaussian(covariance):
            precision = np.linalg.inv(covariance)
            sqrt_precision = np.linalg.cholesky(precision)
            _det = np.linalg.det(covariance)
            p = covariance.shape[0]
            _const = np.sqrt((2 * np.pi) ** p * _det)
            density = lambda x: np.exp(-(x * precision.dot(x)).sum() / 2) / _const
            grad_negative_log_density = lambda x: precision.dot(x)
            sampler = lambda size: sqrt_precision.dot(np.random.standard_normal((p,) + size))
            return randomization.__init__(self,(p,), density, grad_negative_log_density, sampler,
                                 lipschitz=np.linalg.svd(precision)[1].max())

        gaussian(self.covariance)
        #randomization.__init__(self, 1, density, grad_negative_log_density, sampler)


class splitJT(randomization):

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
        sqrt_precision = np.linalg.cholesky(precision)
        _det = np.linalg.det(covariance)
        p = covariance.shape[0]
        _const = np.sqrt((2*np.pi)**p * _det)
        self._density = lambda x: np.exp(-(x * precision.dot(x)).sum() / 2) / _const
        self._grad_negative_log_density = lambda x: precision.dot(x)
        self._sampler = lambda size: sqrt_precision.dot(np.random.standard_normal((p,) + size))
        self.lipschitz = np.linalg.svd(precision)[1].max()

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
        m, n = self.subsample_size, self.total_size # shorthand
        idx = np.zeros(n, np.bool)
        idx[:m] = 1
        np.random.shuffle(idx)

        randomized_loss = loss.subsample(idx)
        randomized_loss.coef *= inv_frac

        return randomized_loss
