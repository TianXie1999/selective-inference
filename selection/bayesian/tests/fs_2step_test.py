from __future__ import print_function
import time

import numpy as np
from scipy.optimize import minimize
from selection.tests.instance import gaussian_instance
from selection.randomized.api import randomization
from selection.bayesian.forward_step_reparametrized import cube_subproblem_fs_linear, cube_objective_fs_linear
from selection.bayesian.forward_stepwise_2steps import selection_probability_objective_fs_2steps

def sel_prob_fs_2steps():
    n = 50
    p = 10
    s = 5
    snr = 5

    X_1, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
    random_Z_1 = np.random.standard_normal(p)
    random_obs_1 = X_1.T.dot(y) + random_Z_1

    active_index_1 = np.argmax(random_obs_1)
    active_1 = np.zeros(p, bool)
    active_1[active_index_1] = 1
    active_sign_1 = np.sign(random_obs_1[active_index_1])
    print("first step--chosen index and sign", active_index_1, active_sign_1)

    X_step2 = X_1[:,~active_1]
    random_Z_2 = np.random.standard_normal(p-1)
    random_obs_2 = X_step2.T.dot(y) + random_Z_2

    active_index_2 = np.argmax(random_obs_2)
    active_2 = np.zeros(p-1, bool)
    active_2[active_index_2] = 1
    active_sign_2 = np.sign(random_obs_2[active_index_2])
    print("second step--chosen index and sign", active_index_2, active_sign_2)

    active = np.zeros(p, bool)
    active[active_index_1] = 1
    (active[~active])[active_index_2] = 1

    feasible_point = np.fabs(np.append(random_obs_1[active_index_1], random_obs_2[active_index_2]))
    nactive = 2

    test_point_primal = np.append(np.append(np.random.uniform(low=-2.0, high=2.0, size=n), 2), 3)
    randomizer = randomization.isotropic_gaussian((p,), 1.)

    parameter = np.random.standard_normal(nactive)
    mean = X_1[:, active].dot(parameter)

    fs = selection_probability_objective_fs_2steps(X_1,
                                                   feasible_point,
                                                   active_1,
                                                   active_2,
                                                   active_sign_1,
                                                   active_sign_2,
                                                   mean,  # in R^n
                                                   noise_variance,
                                                   randomizer)

    toc = time.time()
    sel_prob_fs = fs.minimize2(nstep=100)[::-1]
    tic = time.time()

    print("selection prob and minimizer- fs", sel_prob_fs[0], sel_prob_fs[1])

sel_prob_fs_2steps()