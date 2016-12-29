import time
import numpy as np
import regreg.api as rr
from selection.bayesian.selection_probability_rr import nonnegative_softmax_scaled
from scipy.stats import norm
from selection.randomized.M_estimator import M_estimator
from selection.randomized.glm import pairs_bootstrap_glm, bootstrap_cov
import selection.tests.reports as reports

def myround(a, decimals=1):
    a_x = np.round(a, decimals=1)* 10.
    rem = np.zeros(a.shape[0], bool)
    rem[(np.remainder(a_x, 2) == 1)] = 1
    a_x[rem] = a_x[rem] + 1.
    return a_x/10.


class neg_log_cube_probability(rr.smooth_atom):
    def __init__(self,
                 q, #equals p - E in our case
                 lagrange,
                 randomization_scale = 1., #equals the randomization variance in our case
                 coef=1.,
                 offset=None,
                 quadratic=None):

        self.randomization_scale = randomization_scale
        self.lagrange = lagrange
        self.q = q

        rr.smooth_atom.__init__(self,
                                (self.q,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=None,
                                coef=coef)

    def smooth_objective(self, arg, mode='both', check_feasibility=False, tol=1.e-6):

        arg = self.apply_offset(arg)

        arg_u = (arg + self.lagrange)/self.randomization_scale
        arg_l = (arg - self.lagrange)/self.randomization_scale
        prod_arg = np.exp(-(2. * self.lagrange * arg)/(self.randomization_scale**2))
        neg_prod_arg = np.exp((2. * self.lagrange * arg)/(self.randomization_scale**2))
        cube_prob = norm.cdf(arg_u) - norm.cdf(arg_l)
        log_cube_prob = -np.log(cube_prob).sum()
        threshold = 10 ** -10
        indicator = np.zeros(self.q, bool)
        indicator[(cube_prob > threshold)] = 1
        positive_arg = np.zeros(self.q, bool)
        positive_arg[(arg>0)] = 1
        pos_index = np.logical_and(positive_arg, ~indicator)
        neg_index = np.logical_and(~positive_arg, ~indicator)
        log_cube_grad = np.zeros(self.q)
        log_cube_grad[indicator] = (np.true_divide(-norm.pdf(arg_u[indicator]) + norm.pdf(arg_l[indicator]),
                                        cube_prob[indicator]))/self.randomization_scale

        log_cube_grad[pos_index] = ((-1. + prod_arg[pos_index])/
                                     ((prod_arg[pos_index]/arg_u[pos_index])-
                                      (1./arg_l[pos_index])))/self.randomization_scale

        log_cube_grad[neg_index] = ((arg_u[neg_index] -(arg_l[neg_index]*neg_prod_arg[neg_index]))
                                    /self.randomization_scale)/(1.- neg_prod_arg[neg_index])


        if mode == 'func':
            return self.scale(log_cube_prob)
        elif mode == 'grad':
            return self.scale(log_cube_grad)
        elif mode == 'both':
            return self.scale(log_cube_prob), self.scale(log_cube_grad)
        else:
            raise ValueError("mode incorrectly specified")


class approximate_conditional_prob_E(rr.smooth_atom):

    def __init__(self,
                 t, #point at which density is to computed
                 approx_density,
                 coef = 1.,
                 offset= None,
                 quadratic= None):

        self.t = t
        self.AD = approx_density
        self.q = self.AD.p - self.AD.nactive
        self.inactive_conjugate = self.active_conjugate = approx_density.randomization.CGF_conjugate

        if self.active_conjugate is None:
            raise ValueError(
                'randomization must know its CGF_conjugate -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        lagrange = []
        for key, value in self.AD.penalty.weights.iteritems():
            lagrange.append(value)
        lagrange = np.asarray(lagrange)

        self.inactive_lagrange = lagrange[~self.AD._overall]
        self.active_lagrange = lagrange[self.AD._overall]

        rr.smooth_atom.__init__(self,
                                (self.AD.nactive,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=self.AD.feasible_point,
                                coef=coef)

        self.coefs[:] = self.AD.feasible_point
        self.B_active = self.AD.opt_linear_term[:self.AD.nactive, :self.AD.nactive]
        self.B_inactive = self.AD.opt_linear_term[self.AD.nactive:, :self.AD.nactive]

        self.nonnegative_barrier = nonnegative_softmax_scaled(self.AD.nactive)


    def sel_prob_smooth_objective(self, param, j, mode='both', check_feasibility=False):

        param = self.apply_offset(param)
        index = np.zeros(self.AD.nactive, bool)
        index[j] = 1
        A_j = np.dot(self.AD.score_linear_term, self.AD.Sigma_DT[:,j])/self.AD.Sigma_T[j,j]
        data = np.squeeze(self.t *  A_j)
        null_statistic = self.AD.score_linear_term.dot(self.AD.observed_score_state)-A_j * self.AD.target_observed[j]

        offset_active = self.AD.opt_affine_term[:self.AD.nactive] + null_statistic[:self.AD.nactive] + data[:self.AD.nactive]

        offset_inactive = null_statistic[self.AD.nactive:] + data[self.AD.nactive:]

        active_conj_loss = rr.affine_smooth(self.active_conjugate,
                                            rr.affine_transform(self.B_active, offset_active))

        cube_obj = neg_log_cube_probability(self.q, self.inactive_lagrange, randomization_scale = 1.)

        cube_loss = rr.affine_smooth(cube_obj, rr.affine_transform(self.B_inactive, offset_inactive))

        total_loss = rr.smooth_sum([active_conj_loss,
                                    cube_loss,
                                    self.nonnegative_barrier])

        if mode == 'func':
            f = total_loss.smooth_objective(param, 'func')
            return self.scale(f)
        elif mode == 'grad':
            g = total_loss.smooth_objective(param, 'grad')
            return self.scale(g)
        elif mode == 'both':
            f, g = total_loss.smooth_objective(param, 'both')
            return self.scale(f), self.scale(g)
        else:
            raise ValueError("mode incorrectly specified")

    def minimize2(self, j, step=1, nstep=30, tol=1.e-6):

        current = self.coefs
        current_value = np.inf

        objective = lambda u: self.sel_prob_smooth_objective(u, j, 'func')
        grad = lambda u: self.sel_prob_smooth_objective(u, j, 'grad')

        for itercount in range(nstep):
            newton_step = grad(current)

            # make sure proposal is feasible

            count = 0
            while True:
                count += 1
                proposal = current - step * newton_step
                #print("current proposal and grad", proposal, newton_step)
                if np.all(proposal > 0):
                    break
                step *= 0.5
                if count >= 40:
                    #print(proposal)
                    raise ValueError('not finding a feasible point')

            # make sure proposal is a descent

            count = 0
            while True:
                proposal = current - step * newton_step
                proposed_value = objective(proposal)
                #print(current_value, proposed_value, 'minimize')
                if proposed_value <= current_value:
                    break
                step *= 0.5

            # stop if relative decrease is small

            if np.fabs(current_value - proposed_value) < tol * np.fabs(current_value):
                current = proposal
                current_value = proposed_value
                break

            current = proposal
            current_value = proposed_value

            if itercount % 4 == 0:
                step *= 2

        # print('iter', itercount)
        value = objective(current)

        return current, value


class target_class(object):
    def __init__(self, target_cov):
        self.target_cov = target_cov
        self.shape = target_cov.shape

class approximate_conditional_density_E(rr.smooth_atom, M_estimator):

    def __init__(self, loss, epsilon, penalty, randomization,
                 coef=1.,
                 offset=None,
                 quadratic=None,
                 nstep=10):

        M_estimator.__init__(self, loss, epsilon, penalty, randomization)

        rr.smooth_atom.__init__(self,
                                (1,),
                                offset=offset,
                                quadratic=quadratic,
                                coef=coef)

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
        Sigma_Tinv = np.linalg.inv(Sigma_T)

        score_linear_term = self.score_transform[0]
        (self.opt_linear_term, self.opt_affine_term) = self.opt_transform

        # decomposition
        #print(self.opt_affine_term[nactive:])
        #target_linear_term = (score_linear_term.dot(Sigma_DT)).dot(Sigma_Tinv)
        (self.score_linear_term, self.Sigma_DT, self.Sigma_T) = (score_linear_term, Sigma_DT, Sigma_T)

        # observed target and null statistic
        target_observed = self.observed_score_state[:nactive]
        self.target = target_class(Sigma_T)
        #null_statistic = (score_linear_term.dot(self.observed_score_state))-(target_linear_term.dot(target_observed))

        #(self.target_linear_term, self.target_observed, self.null_statistic) \
        #    = (target_linear_term, target_observed, null_statistic)
        self.target_observed = target_observed
        self.nactive = nactive

        #defining the grid on which marginal conditional densities will be evaluated
        grid_length = 201
        self.grid = np.linspace(-5, 15, num=grid_length)
        #s_obs = np.round(self.target_observed, decimals =1)

        print("observed values", target_observed)
        self.ind_obs = np.zeros(nactive, int)
        self.norm = np.zeros(nactive)
        self.h_approx = np.zeros((nactive, self.grid.shape[0]))

        for j in range(nactive):
            obs = target_observed[j]
            self.norm[j] = Sigma_T[j,j]
            if obs < self.grid[0]:
                self.ind_obs[j] = 0
            elif obs > np.max(self.grid):
                self.ind_obs[j] = grid_length-1
            else:
                self.ind_obs[j] = np.argmin(np.abs(self.grid-obs))

                #self.ind_obs[j] = (np.where(self.grid == obs)[0])[0]
            self.h_approx[j, :] = self.approx_conditional_prob(j)


    def approx_conditional_prob(self, j):
        h_hat = []

        for i in range(self.grid.shape[0]):

            approx = approximate_conditional_prob_E(self.grid[i], self)
            h_hat.append(-(approx.minimize2(j, nstep=50)[::-1])[0])

        return np.array(h_hat)

    def area_normalized_density(self, j, mean):

        normalizer = 0.

        approx_nonnormalized = []
        for i in range(self.grid.shape[0]):
            approx_density = np.exp(-np.true_divide((self.grid[i] - mean) ** 2, 2 * self.norm[j])
                                    + (self.h_approx[j,:])[i])

            normalizer += approx_density

            approx_nonnormalized.append(approx_density)

        return np.cumsum(np.array(approx_nonnormalized / normalizer))

    def approximate_ci(self, j):

        param_grid = np.linspace(-5, 15, num=201)

        area = np.zeros(param_grid.shape[0])

        for k in range(param_grid.shape[0]):

            area_vec = self.area_normalized_density(j, param_grid[k])
            area[k] = area_vec[self.ind_obs[j]]

        region = param_grid[(area >= 0.05) & (area <= 0.95)]

        if region.size > 0:
            return np.nanmin(region), np.nanmax(region)
        else:
            return 0, 0

    def approximate_pvalue(self, j, param):

        area_vec = self.area_normalized_density(j, param)
        area = area_vec[self.ind_obs[j]]

        return 2*min(area, 1-area)



