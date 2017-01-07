import numpy as np
from selection.randomized.M_estimator import M_estimator
from selection.randomized.glm import pairs_bootstrap_glm, bootstrap_cov

from selection.randomized.threshold_score import threshold_score

class target_class(object):
    def __init__(self, target_cov):
        self.target_cov = target_cov
        self.shape = target_cov.shape


class M_estimator_approx(M_estimator):

    def __init__(self, loss, epsilon, penalty, randomization, randomizer):
        M_estimator.__init__(self, loss, epsilon, penalty, randomization)
        self.randomizer = randomizer

    def solve_approx(self):

        self.Msolve()
        self.feasible_point = np.abs(self.initial_soln[self._overall])
        X, _ = self.loss.data
        n, p = X.shape
        self.p = p
        bootstrap_score = pairs_bootstrap_glm(self.loss,
                                              self._overall,
                                              beta_full=self._beta_full,
                                              inactive=~self._overall)[0]

        score_cov = bootstrap_cov(lambda: np.random.choice(n, size=(n,), replace=True), bootstrap_score)

        nactive = self._overall.sum()

        self.score_target_cov = score_cov[:, :nactive]
        self.target_cov = score_cov[:nactive, :nactive]

        self.score_linear_term = self.score_transform[0]
        (self.opt_linear_term, self.opt_affine_term) = self.opt_transform

        self.target_observed = self.observed_score_state[:nactive]
        self.target = target_class(self.target_cov)  # used for naive intervals and naive pvalue
        self.nactive = nactive

        lagrange = []
        for key, value in self.penalty.weights.iteritems():
            lagrange.append(value)
        self.lagrange = np.asarray(lagrange)

        self.B_active = self.opt_linear_term[:nactive, :nactive]
        self.B_inactive = self.opt_linear_term[nactive:, :nactive]


    def setup_map(self, j):

        self.A = np.dot(self.score_linear_term, self.score_target_cov[:, j]) / self.target_cov[j, j]
        self.null_statistic = self.score_linear_term.dot(self.observed_score_state) - self.A * self.target_observed[j]

        self.offset_active = self.opt_affine_term[:self.nactive] + self.null_statistic[:self.nactive]
        self.offset_inactive = self.null_statistic[self.nactive:]



class threshold_score_approx(threshold_score):
    def __init__(self, loss, threshold, randomization, active, inactive, beta_active=None,
                 solve_args={'min_its': 50, 'tol': 1.e-10}):

        threshold_score.__init__(self, loss, threshold, randomization, active, inactive, beta_active=None,
                                 solve_args={'min_its': 50, 'tol': 1.e-10})










