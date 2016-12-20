from __future__ import print_function
import numpy as np
import time
import regreg.api as rr
import selection.tests.reports as reports
from selection.tests.instance import logistic_instance, gaussian_instance
from selection.bayesian.ci_via_approx_density import approximate_conditional_density_E
from selection.tests.flags import SMALL_SAMPLES, SET_SEED
from selection.tests.decorators import wait_for_return_value, register_report, set_sampling_params_iftrue


@register_report(['cover', 'ci_length_clt'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@wait_for_return_value()
def test_approximate_ci_E(n=200, p=10, s=3, snr=5, rho=0,
                          lam_frac=1.,
                          loss='logistic',
                          randomizer='gaussian'):

    from selection.api import randomization

    if loss == "gaussian":
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, snr=snr, sigma=1.)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        loss = rr.glm.gaussian(X, y)
    elif loss == "logistic":
        X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, snr=snr)
        loss = rr.glm.logistic(X, y)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))

    epsilon = 1. / np.sqrt(n)

    W = np.ones(p) * lam
    # W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)
    if randomization=='gaussian':
        randomization = randomization.isotropic_gaussian((p,), scale=1.)
    elif randomizer=='laplace':
        randomization = randomization.laplace((p,), scale=1.)

    ci = approximate_conditional_density_E(loss, epsilon, penalty, randomization)

    ci.solve_approx()
    print("nactive", ci._overall.sum())
    active_set = np.asarray([i for i in range(p) if ci._overall[i]])

    true_support = np.asarray([i for i in range(p) if i < s])

    nactive = ci.nactive

    print("active set, true_support", active_set, true_support)
    active = ci._overall
    #truth = np.round((np.linalg.pinv(X[:, active])).dot(X[:, active].dot(true_beta[active])))
    true_vec = beta[active]

    print("true coefficients", true_vec)

    if (set(active_set).intersection(set(true_support)) == set(true_support))== True:

        ci_active_E = np.zeros((nactive, 2))
        covered = np.zeros(nactive, np.bool)
        ci_length = np.zeros(nactive)
        toc = time.time()
        for j in range(nactive):
            ci_active_E[j, :] = np.array(ci.approximate_ci(j))
            if (ci_active_E[j, 0] <= true_vec[j]) and (ci_active_E[j,1] >= true_vec[j]):
                covered[j] = 1
            ci_length[j] = ci_active_E[j,1] - ci_active_E[j,0]
            print(ci_active_E[j, :])
        tic = time.time()
        print('ci time now', tic - toc)


        return covered, ci_length

    #else:
    #    return 0


def report(niter=50, **kwargs):

    kwargs = {'s': 0, 'n': 200, 'p': 10, 'snr': 7, 'loss':'logistic', 'randomizer':'gaussian'}
    split_report = reports.reports['test_approximate_ci_E']
    screened_results = reports.collect_multiple_runs(split_report['test'],
                                                     split_report['columns'],
                                                     niter,
                                                     reports.summarize_all,
                                                     **kwargs)

    #fig = reports.boot_clt_plot(screened_results, inactive=True, active=False)
    #fig.savefig('multiple_queries_CI.pdf') # will have both bootstrap and CLT on plot


if __name__=='__main__':
    report()
