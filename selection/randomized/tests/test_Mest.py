import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.randomized.randomization import base
from selection.randomized.M_estimator import M_estimator
from selection.randomized.glm_boot import pairs_bootstrap_glm, bootstrap_cov

from selection.algorithms.randomized import logistic_instance
from selection.distributions.discrete_family import discrete_family
from selection.sampling.langevin import projected_langevin

def test_overall_null_two_views():
    s, n, p = 5, 200, 20 

    randomization = base.laplace((p,), scale=0.5)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0.1, snr=7)

    nonzero = np.where(beta)[0]
    lam_frac = 1.

    loss = rr.glm.logistic(X, y)
    epsilon = 1.

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    W = np.ones(p)*lam
    W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)
    # first randomization

    M_est1 = M_estimator(loss, epsilon, penalty, randomization)
    M_est1.solve()
    M_est1.setup_sampler()
    bootstrap_score1 = pairs_bootstrap_glm(M_est1.loss, 
                                           M_est1.overall, 
                                           beta_full=M_est1._beta_full, # this is private -- we "shouldn't" observe this
                                           inactive=M_est1.inactive)[0]

    # second randomization

    M_est2 = M_estimator(loss, epsilon, penalty, randomization)
    M_est2.solve()
    M_est2.setup_sampler()
    bootstrap_score2 = pairs_bootstrap_glm(M_est2.loss, 
                                           M_est2.overall, 
                                           beta_full=M_est2._beta_full, # this is private -- we "shouldn't" observe this
                                           inactive=M_est2.inactive)[0]

    # we take target to be union of two active sets

    active = M_est1.overall + M_est2.overall

    if set(nonzero).issubset(np.nonzero(active)[0]):
        boot_target, target_observed = pairs_bootstrap_glm(loss, active)

        # target are all true null coefficients selected

        target_cov, cov1, cov2 = bootstrap_cov((n, n), boot_target, cross_terms=(bootstrap_score1, bootstrap_score2))

        active_set = np.nonzero(active)[0]
        inactive_selected = I = [i for i in np.arange(active_set.shape[0]) if active_set[i] not in nonzero]

        # is it enough only to bootstrap the inactive ones?
        # seems so...

        A1, b1 = M_est1.condition(cov1[I], target_cov[I][:,I], target_observed[I])
        A2, b2 = M_est2.condition(cov2[I], target_cov[I][:,I], target_observed[I])

        target_inv_cov = np.linalg.inv(target_cov[I][:,I])

        initial_state = np.hstack([target_observed[I],
                                   M_est1.observed_opt_state,
                                   M_est2.observed_opt_state])

        ntarget = len(I)
        target_slice = slice(0, ntarget)
        opt_slice1 = slice(ntarget, p + ntarget)
        opt_slice2 = slice(p + ntarget, 2*p + ntarget)

        def target_gradient(state):
            # with many samplers, we will add up the `target_slice` component
            # many target_grads
            # and only once do the Gaussian addition of full_grad

            target = state[target_slice]
            opt_state1 = state[opt_slice1]
            opt_state2 = state[opt_slice2]
            target_grad1 = M_est1.gradient(target, (A1, b1), opt_state1)
            target_grad2 = M_est2.gradient(target, (A2, b2), opt_state2)

            full_grad = np.zeros_like(state)
            full_grad[opt_slice1] = -target_grad1[1]
            full_grad[opt_slice2] = -target_grad2[1]
            full_grad[target_slice] -= target_grad1[0] + target_grad2[0]
            full_grad[target_slice] -= target_inv_cov.dot(target)

            return full_grad

        def target_projection(state):
            opt_state1 = state[opt_slice1]
            state[opt_slice1] = M_est1.projection(opt_state1)
            opt_state2 = state[opt_slice2]
            state[opt_slice2] = M_est2.projection(opt_state2)
            return state

        target_langevin = projected_langevin(initial_state,
                                             target_gradient,
                                             target_projection,
                                             .5 / (2*p + 1))


        Langevin_steps = 20000
        burning = 10000
        samples = []
        for i in range(Langevin_steps):
            if (i>=burning):
                target_langevin.next()
                samples.append(target_langevin.state[target_slice].copy())
                
        test_stat = lambda x: np.linalg.norm(x)
        observed = test_stat(target_observed[I])
        sample_test_stat = np.array([test_stat(x) for x in samples])

        family = discrete_family(sample_test_stat, np.ones_like(sample_test_stat))
        pval = family.ccdf(0, observed)
        return pval

def test_one_inactive_coordinate(seed=None):
    s, n, p = 5, 200, 20 

    randomization = base.laplace((p,), scale=1.)
    if seed is not None:
        np.random.seed(seed)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0.1, snr=7)

    nonzero = np.where(beta)[0]
    lam_frac = 1.

    loss = rr.glm.logistic(X, y)
    epsilon = 1.

    if seed is not None:
        np.random.seed(seed)
    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    W = np.ones(p)*lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    print lam
    # our randomization

    np.random.seed(seed)
    M_est1 = M_estimator(loss, epsilon, penalty, randomization)
    M_est1.solve()
    M_est1.setup_sampler()
    bootstrap_score1 = pairs_bootstrap_glm(M_est1.loss, 
                                           M_est1.overall, 
                                           beta_full=M_est1._beta_full, # this is private -- we "shouldn't" observe this
                                           inactive=M_est1.inactive)[0]

    active = M_est1.overall
    if set(nonzero).issubset(np.nonzero(active)[0]):
        boot_target, target_observed = pairs_bootstrap_glm(loss, active)

        # target are all true null coefficients selected

        if seed is not None:
            np.random.seed(seed)
        target_cov, cov1 = bootstrap_cov((n, n), boot_target, cross_terms=(bootstrap_score1,))

        # have checked that covariance up to here agrees with other test_glm_langevin example

        active_set = np.nonzero(active)[0]
        inactive_selected = I = [i for i in np.arange(active_set.shape[0]) if active_set[i] not in nonzero]

        # is it enough only to bootstrap the inactive ones?
        # seems so...

        if not I:
            return None, None

        # take the first inactive one
        I = I[:1]
        A1, b1 = M_est1.condition(cov1[I], target_cov[I][:,I], target_observed[I])

        print I, 'I', target_observed[I]
        target_inv_cov = np.linalg.inv(target_cov[I][:,I])

        initial_state = np.hstack([target_observed[I],
                                   M_est1.observed_opt_state])

        ntarget = len(I)
        target_slice = slice(0, ntarget)
        opt_slice1 = slice(ntarget, p + ntarget)

        def target_gradient(state):
            # with many samplers, we will add up the `target_slice` component
            # many target_grads
            # and only once do the Gaussian addition of full_grad

            target = state[target_slice]
            opt_state1 = state[opt_slice1]
            target_grad1 = M_est1.gradient(target, (A1, b1), opt_state1)

            full_grad = np.zeros_like(state)
            full_grad[opt_slice1] = -target_grad1[1]
            full_grad[target_slice] -= target_grad1[0] 
            full_grad[target_slice] -= target_inv_cov.dot(target)

            return full_grad

        def target_projection(state):
            opt_state1 = state[opt_slice1]
            state[opt_slice1] = M_est1.projection(opt_state1)
            return state

        target_langevin = projected_langevin(initial_state,
                                             target_gradient,
                                             target_projection,
                                             1. / p)


        Langevin_steps = 10000
        burning = 2000
        samples = []
        for i in range(Langevin_steps + burning):
            if (i>burning):
                target_langevin.next()
                samples.append(target_langevin.state[target_slice].copy())
                
        test_stat = lambda x: x
        observed = test_stat(target_observed[I])
        sample_test_stat = np.array([test_stat(x) for x in samples])

        family = discrete_family(sample_test_stat, np.ones_like(sample_test_stat))
        pval = family.ccdf(0, observed)
        pval = 2 * min(pval, 1-pval)
        
        _i = I[0]
        naive_Z = target_observed[_i] / np.sqrt(target_cov[_i,_i])
        naive_pval = ndist.sf(np.fabs(naive_Z))
        return pval, naive_pval
    else:
        return None, None