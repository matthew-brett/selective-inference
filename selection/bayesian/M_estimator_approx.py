import numpy as np
from selection.randomized.M_estimator import M_estimator
from selection.randomized.glm import pairs_bootstrap_glm, bootstrap_cov


class target_class(object):
    def __init__(self, target_cov):
        self.target_cov = target_cov
        self.shape = target_cov.shape



class M_estimator_approx(M_estimator):

    def __init__(self, loss, epsilon, penalty, randomization):
        M_estimator.__init__(self, loss, epsilon, penalty, randomization)

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

        Sigma_DT = score_cov[:, :nactive]
        Sigma_T = score_cov[:nactive, :nactive]

        score_linear_term = self.score_transform[0]
        (self.opt_linear_term, self.opt_affine_term) = self.opt_transform

        (self.score_linear_term, self.Sigma_DT, self.Sigma_T) = (score_linear_term, Sigma_DT, Sigma_T)

        # observed target and null statistic
        target_observed = self.observed_score_state[:nactive]
        self.target = target_class(Sigma_T)

        self.target_observed = target_observed
        self.nactive = nactive

        lagrange = []
        for key, value in self.penalty.weights.iteritems():
            lagrange.append(value)
        self.lagrange = np.asarray(lagrange)

        self.B_active = self.opt_linear_term[:nactive, :nactive]
        self.B_inactive = self.opt_linear_term[nactive:, :nactive]


    def setup_map(self, j):

        self.A = np.dot(self.score_linear_term, self.Sigma_DT[:, j]) / self.Sigma_T[j, j]
        #data = np.squeeze(self.t * A_j)
        self.null_statistic = self.score_linear_term.dot(self.observed_score_state) - A * self.target_observed[j]

        self.offset_active = self.opt_affine_term[:self.nactive] + self.null_statistic[:self.nactive]

        self.offset_inactive = self.null_statistic[self.nactive:]














