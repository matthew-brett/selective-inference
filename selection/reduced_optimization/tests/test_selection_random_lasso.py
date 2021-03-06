from __future__ import print_function
import numpy as np
import time
import regreg.api as rr
from selection.reduced_optimization.initial_soln import selection
from selection.tests.instance import logistic_instance, gaussian_instance

from ..par_random_lasso_reduced import (selection_probability_random_lasso, 
                                        sel_inf_random_lasso)
from ..estimator import M_estimator_approx
from selection.api import randomization

def test_selection():
    n = 500
    p = 100
    s = 0
    signal = 0.

    np.random.seed(3)  # ensures different y
    X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, sigma=1., rho=0, signal=signal)
    lam = 1. * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma

    n, p = X.shape

    loss = rr.glm.gaussian(X, y)
    epsilon = 1. / np.sqrt(n)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),weights=dict(zip(np.arange(p), W)), lagrange=1.)
    randomizer = randomization.isotropic_gaussian((p,), scale=1.)

    M_est = M_estimator_approx(loss, epsilon, penalty, randomizer, 'gaussian', 'parametric')
    M_est.solve_approx()
    active = M_est._overall
    active_set = np.asarray([i for i in range(p) if active[i]])
    nactive = np.sum(active)

    prior_variance = 1000.
    noise_variance = sigma ** 2

    generative_mean = np.zeros(p)
    generative_mean[:nactive] = M_est.initial_soln[active]
    sel_split = selection_probability_random_lasso(M_est, generative_mean)
    min = sel_split.minimize2(nstep=200)
    print(min[0], min[1])

    test_point = np.append(M_est.observed_score_state, np.abs(M_est.initial_soln[M_est._overall]))
    print("value of likelihood", sel_split.likelihood_loss.smooth_objective(test_point, mode= "func"))

    inv_cov = np.linalg.inv(M_est.score_cov)
    lik = (M_est.observed_score_state-generative_mean).T.dot(inv_cov).dot(M_est.observed_score_state-generative_mean)/2.
    print("value of likelihood check", lik)
    grad = inv_cov.dot(M_est.observed_score_state-generative_mean)
    print("grad at likelihood loss", grad)



