from __future__ import print_function
import numpy as np
import regreg.api as rr
import selection.tests.reports as reports
from selection.tests.instance import logistic_instance, gaussian_instance
from selection.approx_ci.ci_via_approx_density import approximate_conditional_density
from selection.approx_ci.estimator_approx import M_estimator_approx

from selection.tests.flags import SMALL_SAMPLES, SET_SEED
from selection.tests.decorators import wait_for_return_value, register_report, set_sampling_params_iftrue
from selection.randomized.query import naive_confidence_intervals
from selection.randomized.query import naive_pvalues


@register_report(['cover', 'ci_length', 'truth', 'naive_cover', 'naive_pvalues', 'ci_length_naive'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@wait_for_return_value()
def test_glm(n=1000,
             p=500,
             s=0,
             snr=3,
             rho=0.,
             lam_frac = 3.2,
             loss='logistic',
             randomizer='gaussian'):

    from selection.api import randomization
    if snr == 0:
        s=0
    if loss == "gaussian":
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, snr=snr, sigma=1.)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        loss = rr.glm.gaussian(X, y)
    elif loss == "logistic":
        X, y, beta, nonzero = logistic_instance(n=n, p=p, s=s, rho=rho, snr=snr)
        loss = rr.glm.logistic(X, y)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))

    true_support = nonzero
    epsilon = 1. / np.sqrt(n)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)
    if randomizer=='gaussian':
        randomization = randomization.isotropic_gaussian((p,), scale=1.)
    elif randomizer=='laplace':
        randomization = randomization.laplace((p,), scale=1.)

    M_est = M_estimator_approx(loss, epsilon, penalty, randomization, randomizer)
    M_est.solve_approx()
    active = M_est._overall
    active_set = np.asarray([i for i in range(p) if active[i]])
    nactive = np.sum(active)

    if nactive==0:
        return None
    print("active set, true_support", active_set, true_support)
    true_vec = beta[active]
    #print("true coefficients", true_vec)

    if (set(active_set).intersection(set(true_support)) == set(true_support))== True:
        ci = approximate_conditional_density(M_est)
        ci.solve_approx()

        ci_sel = np.zeros((nactive, 2))
        sel_covered = np.zeros(nactive, np.bool)
        sel_length = np.zeros(nactive)
        pivots = np.zeros(nactive)

        class target_class(object):
            def __init__(self, target_cov):
                self.target_cov = target_cov
                self.shape = target_cov.shape
        target = target_class(M_est.target_cov)

        ci_naive = naive_confidence_intervals(target, M_est.target_observed)
        naive_pvals = naive_pvalues(target, M_est.target_observed, true_vec)
        naive_covered = np.zeros(nactive)
        naive_length = np.zeros(nactive)

        for j in range(nactive):
            ci_sel[j, :] = np.array(ci.approximate_ci(j))
            if (ci_sel[j, 0] <= true_vec[j]) and (ci_sel[j,1] >= true_vec[j]):
                sel_covered[j] = 1
            sel_length[j] = ci_sel[j,1] - ci_sel[j,0]
            print(ci_sel[j, :])
            pivots[j] = ci.approximate_pvalue(j, true_vec[j])

            # naive ci
            if (ci_naive[j,0]<=true_vec[j]) and (ci_naive[j,1]>=true_vec[j]):
                naive_covered[j]+=1
            naive_length[j] = ci_naive[j,1] - ci_naive[j,0]

        return sel_covered, sel_length, pivots, naive_covered, naive_pvals, naive_length
    #else:
    #    return 0

def report(niter=100, **kwargs):

    kwargs = {'s': 0, 'n': 600, 'p': 100, 'snr': 5, 'loss': 'logistic', 'randomizer':'laplace'}
    split_report = reports.reports['test_glm']
    screened_results = reports.collect_multiple_runs(split_report['test'],
                                                     split_report['columns'],
                                                     niter,
                                                     reports.summarize_all,
                                                     **kwargs)

    fig = reports.pivot_plot_plus_naive(screened_results)
    fig.savefig('approx_pivots_glm.pdf')


if __name__=='__main__':
    report()
