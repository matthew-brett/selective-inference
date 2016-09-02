import numpy as np
from regreg.smooth.glm import glm as regreg_glm
import regreg.api as rr

class group_lasso_sampler(object):

    def __init__(self, loss, epsilon, penalty, randomization, solve_args={'min_its':50, 'tol':1.e-10}):
        """
        Fits the logistic regression to a candidate active set, without penalty.
        Calls the method bootstrap_covariance() to bootstrap the covariance matrix.

        Computes $\bar{\beta}_E$ which is the restricted 
        M-estimator (i.e. subject to the constraint $\beta_{-E}=0$).

        Parameters:
        -----------

        active: np.bool
            The active set from fitting the logistic lasso

        solve_args: dict
            Arguments to be passed to regreg solver.

        Returns:
        --------

        None

        Notes:
        ------

        Sets self._beta_unpenalized which will be used in the covariance matrix calculation.
        Also computes Hessian of loss at restricted M-estimator as well as the bootstrap covariance.

        """

        (self.loss,
         self.epsilon,
         self.penalty,
         self.randomization) = (loss,
                                epsilon,
                                penalty,
                                randomization)
         
        # initial solution

        problem = rr.simple_problem(loss, penalty)
        self._randomZ = self.randomization.sample()
        self._random_term = rr.identity_quadratic(epsilon, 0, -self._randomZ, 0)
        solve_args = {'tol': 1.e-10, 'min_its': 100, 'max_its': 500}

        self.initial_soln = problem.solve(self._random_term, **solve_args)


    def setup_sampler(self, solve_args={'min_its':50, 'tol':1.e-10}):

        (loss,
         epsilon,
         penalty,
         randomization,
         initial_soln) = (self.loss,
                          self.epsilon,
                          self.penalty,
                          self.randomization,
                          self.initial_soln)

        # find the active groups and their direction vectors
        # as well as unpenalized groups

        groups = np.unique(penalty.groups) 
        active_groups = np.zeros(len(groups), np.bool)
        unpenalized_groups = np.zeros(len(groups), np.bool)

        active_directions = []
        active = np.zeros(loss.shape, np.bool)
        unpenalized = np.zeros(loss.shape, np.bool)

        initial_scalings = []

        for i, g in enumerate(groups):
            group = penalty.groups == g
            active_groups[i] = (np.linalg.norm(initial_soln[group]) > 1.e-6 * penalty.weights[g]) and (penalty.weights[g] > 0)
            unpenalized_groups[i] = (penalty.weights[g] == 0)
            if active_groups[i]:
                active[group] = True
                z = np.zeros_like(active)
                z[group] = initial_soln[group] / np.linalg.norm(initial_soln[group])
                active_directions.append(z)
                initial_scalings.append(np.linalg.norm(initial_soln[group]))
            if unpenalized_groups[i]:
                unpenalized[group] = True

        # solve the restricted problem

        overall = active + unpenalized
        inactive = ~overall

        # initial state for opt variables

        initial_subgrad = -(self.loss.smooth_objective(self.initial_soln, 'grad') + self._random_term.smooth_objective(self.initial_soln, 'grad'))
        initial_subgrad = initial_subgrad[inactive]
        initial_unpenalized = self.initial_soln[unpenalized]
        self._initial_state = np.concatenate([initial_scalings,
                                              initial_unpenalized,
                                              initial_subgrad], axis=0)

        active_directions = np.array(active_directions).T


        X, Y = loss.data
        if self._is_transform:
            raise NotImplementedError('to fit restricted model, X must be an ndarray or scipy.sparse; general transforms not implemented')
        X_restricted = X[:,overall]
        loss_restricted = rr.affine_smooth(self.loss, X_restricted)
        _beta_unpenalized = loss_restricted.solve(**solve_args)
        beta_full = np.zeros(active.shape)
        beta_full[active] = _beta_unpenalized
        _hessian = loss.hessian(beta_full)

        # form linear part

        self.num_opt_var = p = loss.shape[0] # shorthand for p

        # (\bar{\beta}_{E \cup U}, N_{-E}, c_E, \beta_U, z_{-E})
        # E for active
        # U for unpenalized
        # -E for inactive

        _opt_linear_term = np.zeros((p, p))
        _score_linear_term = np.zeros((p, active_groups.sum() + unpenalized.sum() + inactive.sum()))

        # \bar{\beta}_{E \cup U} piece -- the unpenalized M estimator

        Mest_slice = slice(0, overall.sum())
        _Mest_hessian = _hessian[:,overall]
        _score_linear_term[:,Mest_slice] = -_Mest_hessian

        # N_{-(E \cup U)} piece -- inactive coordinates of score of M estimator at unpenalized solution

        null_slice = slice(overall.sum(), p)
        _score_linear_term[inactive][:,null_slice] = -np.identity(inactive.sum())

        # c_E piece 

        scaling_slice = slice(0, active_groups.sum())
        _opt_hessian = (_hessian + epsilon * np.identity(p)).dot(active_directions)
        _opt_linear_term[:,scaling_slice] = _opt_hessian

        # beta_U piece

        unpenalized_slice = slice(active_groups.sum(), active_groups.sum() + unpenalized.sum())
        _opt_linear_term[:,unpenalized_slice] = _hessian[:,unpenalized] + epsilon * np.identity(unpenalized.sum())

        # subgrad piece
        subgrad_slice = slice(active_groups.sum() + unpenalized.sum(), active_groups.sum() + inactive.sum() + unpenalized.sum())
        _opt_linear_term[inactive][:,subgrad_slice] = np.identity(inactive.sum())

        # form affine part

        _opt_affine_term = np.zeros(p)
        for i, g in enumerate(groups):
            if active_groups[i]:
                group = penalty.groups == g
                _opt_affine_term[group] = active_directions[i][group] * penalty.weights[g]

        # two transforms that encode score and optimization
        # variable roles 

        # later, conditioning will modify `score_transform`

        self.opt_transform = rr.affine_transform(_opt_linear_term, _opt_affine_term)
        self.score_transform = rr.linear_transform(_score_linear_term)

        # now store everything needed for the projections
        # the projection acts only on the optimization
        # variables

        self.scaling_slice = scaling_slice

        new_groups = penalty.groups[inactive]
        new_weights = [penalty.weights[g] for g in penalty.weights.keys() if g in np.unique(new_groups)]

        # we form a dual group lasso object
        # to do the projection

        self.group_lasso_dual = rr.group_lasso_dual(new_groups, weights=new_weights, bound=1.)
        self.subgrad_slice = subgrad_slice

        self.setup_bootstrap()

        def setup_bootstrap(self):
            """
            Should define a callable _boot_score
            that takes `indices` and returns
            a bootstrap sample of (\bar{\beta}_{E \cup U}, N_{-(E \cup U)})
            """
            # form objects needed to bootstrap the score
            # which will be used to estimate covariance
            # of score and target model parameters

            # this code below will work for GLMs
            # not general M-estimators !!!
            # self.loss is the loss for a saturated GLM likelihood

            # gradient of saturated loss if \mu - Y

            if isinstance(self.loss, logistic_loglike):
                Y = Y[0]

            _boot_mu = lambda X: self.loss.smooth_objective(X.dot(beta_full), 'grad') + Y

            _bootQ = np.zeros((p,p))

            _bootW = np.diag(self.loss.hessian(X.dot(beta_full)))
            _bootQ = X[:, overall].T.dot(self._bootW.dot(X[:, overall]))
            _bootQinv = np.linalg.inv(self._bootQ)
            _bootC = X[:, inactive].T.dot(self._bootW.dot(X[:, active]))
            _bootI = self._bootC.dot(self._bootQinv)

            noverall = overall.sum()
            def _boot_score(X_star, Y_star):
                initial_score = X_star.T.dot(Y_star - _boot_mu(X_star))
                result = np.zeros_like(initial_score)
                result[:noverall] = _bootQinv.dot(initial_score)
                result[noverall:] = initial_score[inactive] + _bootI.dot(result[:noverall])
                return result
            self._boot_score = _boot_score

    def projection(self, opt_state):
        """
        Full projection for Langevin.

        The state here will be only the state of the optimization variables.
        """

        if not hasattr(self, "scaling_slice"):
            raise ValueError('setup_sampler should be called before using this function')

        new_state = opt_state.copy() # not really necessary to copy
        new_state[self.scaling_slice] = np.maximum(opt_state[self.scaling_slice], 0)
        new_state[self.subgrad_slice] = self.group_lasso_dual.bound_prox(state[self.subgrad_slice])

        return new_state

    def gradient(self, data_state, data_transform, opt_state):
        """
        Randomization derivative at full state.
        """

        if not hasattr(self, "opt_transform"):
            raise ValueError('setup_sampler should be called before using this function')

        # omega
        full_state = (data_transform.affine_map(data_state) + 
                      self.opt_transform.affine_map(opt_state))
        randomization_derivative = self.randomization.gradient(full_state)
        data_grad = data_transform.adjoint_map(randomization_derivative)
        opt_grad = self.opt_transform.adjoint_map(randomization_derivative)
        return data_grad, opt_grad

    def bootstrap_score(self, indices):
        """
        """

        if not hasattr(self, "_boot_score"):
            raise ValueError('setup_bootstrap should be called before using this function')

        X, Y = self._bootX, self._bootY
        Y_star = Y[indices]
        X_star = X[indices]

        return self._boot_score(X_star, Y_star)



