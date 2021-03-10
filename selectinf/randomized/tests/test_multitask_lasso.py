import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import norm as ndist

from selectinf.randomized.multitask_lasso import multi_task_lasso
from selectinf.tests.instance import gaussian_multitask_instance

def cross_validate_posi_global(ntask=2,
                   nsamples=500 * np.ones(2),
                   p=100,
                   global_sparsity=.8,
                   task_sparsity=.3,
                   sigma=1. * np.ones(2),
                   signal_fac=0.5,
                   rhos=0. * np.ones(2)):

    nsamples = nsamples.astype(int)

    signal = np.sqrt(signal_fac * 2 * np.log(p))

    response_vars, predictor_vars, beta, _gaussian_noise = gaussian_multitask_instance(ntask,
                                                                                       nsamples,
                                                                                       p,
                                                                                       global_sparsity,
                                                                                       task_sparsity,
                                                                                       sigma,
                                                                                       signal,
                                                                                       rhos,
                                                                                       random_signs=True,
                                                                                       equicorrelated=False)[:4]
    cv_weights = []

    for i in range(10):

        train = np.random.choice(np.arange(nsamples[0]), size=np.int(np.round(.8*nsamples[0])), replace=False)
        test = np.setdiff1d(np.arange(nsamples[0]),train)

        response_vars_train = {i: response_vars[i][train] for i in range(ntask)}
        predictor_vars_train = {i: predictor_vars[i][train] for i in range(ntask)}

        response_vars_test = {i: response_vars[i][test] for i in range(ntask)}
        predictor_vars_test = {i: predictor_vars[i][test] for i in range(ntask)}

        lambdamin = 1.0
        lambdamax = 2*np.sqrt(2*np.log(p))
        weights = np.arange(np.log(lambdamin),np.log(lambdamax), lambdamax/50)
        weights = np.exp(weights)

        errors = []

        for w in range(len(weights)):

            feature_weight = weights[w] * np.ones(p)
            sigmas_ = sigma
            randomizer_scales = .7 * sigmas_

            _initial_omega = np.array([randomizer_scales[i] * _gaussian_noise[(i * p):((i + 1) * p)] for i in range(ntask)]).T

            try:
                multi_lasso = multi_task_lasso.gaussian(predictor_vars_train,
                                            response_vars_train,
                                            feature_weight,
                                            ridge_term=None,
                                            randomizer_scales=randomizer_scales)

                multi_lasso.fit(perturbations=_initial_omega)

                dispersions = sigma ** 2

                estimate, observed_info_mean, _, _, intervals = multi_lasso.multitask_inference_global(dispersions=dispersions)

            except:
                sum = 0
                for j in range(ntask):
                    sum += np.linalg.norm(response_vars_test[j], 2)
                errors.append(sum)

            error = []

            for j in range(ntask):

                error.extend(response_vars_test[j] - (predictor_vars_test[j])[:, multi_lasso.active_global].dot(estimate))

            error = np.sqrt(np.sum(np.square(error))) / (len(test) * ntask)
            errors.append(error)

        idx_min_error = np.int(np.argmin(errors))
        cv_weights.append(weights[idx_min_error])
        print(cv_weights)

    return(np.mean(np.asarray(cv_weights)))


def cross_validate_posi_hetero(ntask=2,
                   nsamples=500 * np.ones(2),
                   p=100,
                   global_sparsity=.8,
                   task_sparsity=.3,
                   sigma=1. * np.ones(2),
                   signal_fac=np.array([1., 5.]),
                   rhos=0. * np.ones(2),
                   randomizer_scale =1):

    nsamples = nsamples.astype(int)

    signal = np.sqrt(signal_fac * 2 * np.log(p))

    response_vars, predictor_vars, beta, _gaussian_noise = gaussian_multitask_instance(ntask,
                                                                                       nsamples,
                                                                                       p,
                                                                                       global_sparsity,
                                                                                       task_sparsity,
                                                                                       sigma,
                                                                                       signal,
                                                                                       rhos,
                                                                                       random_signs=True,
                                                                                       equicorrelated=False)[:4]
    cv_weights = []

    for i in range(10):

        train = np.random.choice(np.arange(nsamples[0]), size=np.int(np.round(.8*nsamples[0])), replace=False)
        test = np.setdiff1d(np.arange(nsamples[0]),train)

        response_vars_train = {i: response_vars[i][train] for i in range(ntask)}
        predictor_vars_train = {i: predictor_vars[i][train] for i in range(ntask)}

        response_vars_test = {i: response_vars[i][test] for i in range(ntask)}
        predictor_vars_test = {i: predictor_vars[i][test] for i in range(ntask)}

        lambdamin = 1.0
        lambdamax = 3*np.sqrt(2*np.log(p))
        weights = np.arange(np.log(lambdamin),np.log(lambdamax), lambdamax/50)
        weights = np.exp(weights)

        errors = []

        for w in range(len(weights)):

            print(w)

            feature_weight = weights[w] * np.ones(p)
            sigmas_ = sigma
            randomizer_scales = randomizer_scale * sigmas_

            _initial_omega = np.array(
                [randomizer_scales[i] * _gaussian_noise[(i * p):((i + 1) * p)] for i in range(ntask)]).T

            try:
                multi_lasso = multi_task_lasso.gaussian(predictor_vars_train,
                                                    response_vars_train,
                                                    feature_weight,
                                                    ridge_term=None,
                                                    randomizer_scales=randomizer_scales)

                multi_lasso.fit(perturbations=_initial_omega)

                active_signs = multi_lasso.fit(perturbations=_initial_omega)

                dispersions = sigma ** 2

                estimate, observed_info_mean, Z_scores, pvalues, intervals = multi_lasso.multitask_inference_hetero(
                dispersions=dispersions)

            except:
                sum = 0
                for j in range(ntask):
                    sum += np.linalg.norm(response_vars_test[j], 2)
                errors.append(sum)
                continue

            error = []

            idx = 0

            for j in range(ntask):

                idx_new = np.sum(active_signs[:, j] != 0)
                if idx_new == 0:
                    continue
                error.extend(response_vars_test[j] - (predictor_vars_test[j])[:, (active_signs[:, j] != 0)].dot(
                    estimate[idx:idx + idx_new]))
                idx = idx + idx_new

            error = np.sqrt(np.sum(np.square(error))) / (len(test) * ntask)
            errors.append(error)

        idx_min_error = np.int(np.argmin(errors))
        cv_weights.append(weights[idx_min_error])
        print(cv_weights)

    return (np.mean(np.asarray(cv_weights)))


def cross_validate_naive_hetero(ntask=2,
                   nsamples=500 * np.ones(2),
                   p=100,
                   global_sparsity=.8,
                   task_sparsity=.3,
                   sigma=1. * np.ones(2),
                   signal_fac=np.array([1., 5.]),
                   rhos=0. * np.ones(2),
                   randomizer_scale =1):

    nsamples = nsamples.astype(int)

    signal = np.sqrt(signal_fac * 2 * np.log(p))

    response_vars, predictor_vars, beta, _gaussian_noise = gaussian_multitask_instance(ntask,
                                                                                       nsamples,
                                                                                       p,
                                                                                       global_sparsity,
                                                                                       task_sparsity,
                                                                                       sigma,
                                                                                       signal,
                                                                                       rhos,
                                                                                       random_signs=True,
                                                                                       equicorrelated=False)[:4]
    cv_weights = []

    for i in range(10):

        train = np.random.choice(np.arange(nsamples[0]), size=np.int(np.round(.8*nsamples[0])), replace=False)
        test = np.setdiff1d(np.arange(nsamples[0]),train)

        response_vars_train = {i: response_vars[i][train] for i in range(ntask)}
        predictor_vars_train = {i: predictor_vars[i][train] for i in range(ntask)}

        response_vars_test = {i: response_vars[i][test] for i in range(ntask)}
        predictor_vars_test = {i: predictor_vars[i][test] for i in range(ntask)}

        lambdamin = .5
        lambdamax = 3*np.sqrt(2*np.log(p))
        weights = np.arange(np.log(lambdamin),np.log(lambdamax), lambdamax/50)
        weights = np.exp(weights)

        errors = []

        for w in range(len(weights)):

            print(errors,"err")

            feature_weight = weights[w] * np.ones(p)
            sigmas_ = sigma
            randomizer_scales = randomizer_scale * sigmas_

            perturbations = np.zeros((p, ntask))

            try:
                multi_lasso = multi_task_lasso.gaussian(predictor_vars_train,
                                            response_vars_train,
                                            feature_weight,
                                            ridge_term=None,
                                            randomizer_scales=randomizer_scales,perturbations=perturbations)

                active_signs = multi_lasso.fit()


            except:
                sum = 0
                for j in range(ntask):
                    sum += np.linalg.norm(response_vars_test[j], 2)
                errors.append(sum)
                continue

            error = []

            idx = 0

            for j in range(ntask):

                idx_new = np.sum(active_signs[:, j] != 0)
                if idx_new ==0:
                    continue
                X, y = multi_lasso.loglikes[j].data
                observed_target = np.linalg.pinv(X[:, (active_signs[:, j] != 0)]).dot(y)
                error.extend(response_vars_test[j] - (predictor_vars_test[j])[:, (active_signs[:, j] != 0)].dot(observed_target))

            error = np.sqrt(np.sum(np.square(error))) / (len(test) * ntask)
            errors.append(error)

        idx_min_error = np.int(np.argmin(errors))
        cv_weights.append(weights[idx_min_error])
        print(cv_weights)

    return(np.mean(np.asarray(cv_weights)))


def test_multitask_lasso_global(ntask=2,
                                nsamples=500 * np.ones(2),
                                p=100,
                                global_sparsity=.8,
                                task_sparsity=.3,
                                sigma=1. * np.ones(2),
                                signal_fac=0.5,
                                rhos=0. * np.ones(2),
                                weight=2.):

    nsamples = nsamples.astype(int)

    signal = np.sqrt(signal_fac * 2 * np.log(p))

    response_vars, predictor_vars, beta, _gaussian_noise = gaussian_multitask_instance(ntask,
                                                                                       nsamples,
                                                                                       p,
                                                                                       global_sparsity,
                                                                                       task_sparsity,
                                                                                       sigma,
                                                                                       signal,
                                                                                       rhos,
                                                                                       random_signs=False,
                                                                                       equicorrelated=False)[:4]

    feature_weight = weight * np.ones(p)

    # sigmas_ = np.array([np.std(response_vars[i]) for i in range(ntask)])
    sigmas_ = sigma

    randomizer_scales = 0.7 * sigmas_

    # ridge_terms = np.array([np.std(response_vars[i]) * np.sqrt(np.mean((predictor_vars[i] ** 2).sum(0)))/ np.sqrt(nsamples[i] - 1)
    #                          for i in range(ntask)])

    _initial_omega = np.array([randomizer_scales[i] * _gaussian_noise[(i * p):((i + 1) * p)] for i in range(ntask)]).T

    multi_lasso = multi_task_lasso.gaussian(predictor_vars,
                                            response_vars,
                                            feature_weight,
                                            ridge_term=None,
                                            randomizer_scales=randomizer_scales)

    multi_lasso.fit(perturbations=_initial_omega)

    # dispersions = np.array([np.linalg.norm(response_vars[i] -
    #                                        predictor_vars[i].dot(np.linalg.pinv(predictor_vars[i]).dot(response_vars[i]))) ** 2 / (nsamples[i] - p)
    #                        for i in range(ntask)])

    dispersions = sigma ** 2

    estimate, observed_info_mean, _, _, intervals = multi_lasso.multitask_inference_global(dispersions=dispersions)

    beta_target_ = []

    for j in range(ntask):
        beta_target_.extend(
            np.linalg.pinv((predictor_vars[j])[:, multi_lasso.active_global]).dot(predictor_vars[j].dot(beta[:, j])))

    beta_target_ = np.asarray(beta_target_)
    beta_target = multi_lasso.W_coef.dot(beta_target_)

    coverage = (beta_target > intervals[:, 0]) * (beta_target <
                                                  intervals[:, 1])

    pivot_ = ndist.cdf((estimate - beta_target) / np.sqrt(np.diag(observed_info_mean)))
    pivot = 2 * np.minimum(pivot_, 1. - pivot_)

    return coverage, intervals[:, 1] - intervals[:, 0], pivot


def test_multitask_lasso_naive_global(ntask=2,
                                      nsamples=500 * np.ones(2),
                                      p=100,
                                      global_sparsity=.8,
                                      task_sparsity=.3,
                                      sigma=1. * np.ones(2),
                                      signal_fac=0.5,
                                      rhos=0. * np.ones(2),
                                      weight=2.):
    nsamples = nsamples.astype(int)
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    while True:
        response_vars, predictor_vars, beta = gaussian_multitask_instance(ntask,
                                                                          nsamples,
                                                                          p,
                                                                          global_sparsity,
                                                                          task_sparsity,
                                                                          sigma,
                                                                          signal,
                                                                          rhos,
                                                                          random_signs=True,
                                                                          equicorrelated=False)[:3]

        feature_weight = weight * np.ones(p)

        sigmas_ = sigma

        perturbations = np.zeros((p, ntask))

        multi_lasso = multi_task_lasso.gaussian(predictor_vars,
                                                response_vars,
                                                feature_weight,
                                                ridge_term=None,
                                                randomizer_scales=1. * sigmas_,
                                                perturbations=perturbations)
        active_signs = multi_lasso.fit()

        dispersions = sigma ** 2

        estimate, observed_info_mean, _, _, intervals = multi_lasso.multitask_inference_global(dispersions=dispersions)

        coverage = []
        pivot = []

        if (active_signs != 0).sum() > 0:

            beta_target_ = []
            observed_target_ = []
            tot_par = multi_lasso.active_global.shape[0]

            prec_target = np.zeros((tot_par, tot_par))
            for j in range(ntask):
                beta_target_.extend(np.linalg.pinv((predictor_vars[j])[:, multi_lasso.active_global]).dot(
                    predictor_vars[j].dot(beta[:, j])))
                Qfeat = np.linalg.inv((predictor_vars[j])[:, multi_lasso.active_global].T.dot(
                    (predictor_vars[j])[:, multi_lasso.active_global]))
                observed_target_.extend(
                    np.linalg.pinv((predictor_vars[j])[:, multi_lasso.active_global]).dot(response_vars[j]))
                prec_target += np.linalg.inv(Qfeat * dispersions[j])

            beta_target_ = np.asarray(beta_target_)
            beta_target = multi_lasso.W_coef.dot(beta_target_)
            observed_target_ = np.asarray(observed_target_)
            observed_target = multi_lasso.W_coef.dot(observed_target_)
            cov_target = np.linalg.inv(prec_target)

            alpha = 1. - 0.90
            quantile = ndist.ppf(1 - alpha / 2.)
            intervals = np.vstack([observed_target - quantile * np.sqrt(np.diag(cov_target)),
                                   observed_target + quantile * np.sqrt(np.diag(cov_target))]).T
            coverage.extend((beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1]))
            pivot_ = ndist.cdf((observed_target - beta_target) / np.sqrt(np.diag(cov_target)))
            pivot.extend(2 * np.minimum(pivot_, 1. - pivot_))

            return np.asarray(coverage), intervals[:, 1] - intervals[:, 0], np.asarray(pivot)


def test_multitask_lasso_hetero(ntask=2,
                                nsamples=500 * np.ones(2),
                                p=100,
                                global_sparsity=.8,
                                task_sparsity=.3,
                                sigma=1. * np.ones(2),
                                signal_fac= np.array([1., 5.]),
                                rhos=0. * np.ones(2),
                                weight=2.,
                                randomizer_scale=1):
    nsamples = nsamples.astype(int)

    signal = np.sqrt(signal_fac * 2 * np.log(p))

    while True:

        response_vars, predictor_vars, beta, _gaussian_noise = gaussian_multitask_instance(ntask,
                                                                                           nsamples,
                                                                                           p,
                                                                                           global_sparsity,
                                                                                           task_sparsity,
                                                                                           sigma,
                                                                                           signal,
                                                                                           rhos,
                                                                                           random_signs=True,
                                                                                           equicorrelated=False)[:4]

        feature_weight = weight * np.ones(p)

        sigmas_ = sigma
        randomizer_scales = randomizer_scale * np.array([sigmas_[i] for i in range(ntask)])

        _initial_omega = np.array(
            [randomizer_scales[i] * _gaussian_noise[(i * p):((i + 1) * p)] for i in range(ntask)]).T

        multi_lasso = multi_task_lasso.gaussian(predictor_vars,
                                                response_vars,
                                                feature_weight,
                                                ridge_term=None,
                                                randomizer_scales=randomizer_scales,
                                                perturbations=None)

        active_signs = multi_lasso.fit(perturbations=_initial_omega)

        if (active_signs != 0).sum() > 0:

            dispersions = sigma ** 2

            estimate, observed_info_mean, Z_scores, pvalues, intervals = multi_lasso.multitask_inference_hetero(
                dispersions=dispersions)

            beta_target = []

            for i in range(ntask):
                X, y = multi_lasso.loglikes[i].data
                beta_target.extend(np.linalg.pinv(X[:, (active_signs[:, i] != 0)]).dot(X.dot(beta[:, i])))

            beta_target = np.asarray(beta_target)
            pivot_ = ndist.cdf((estimate - beta_target) / np.sqrt(np.diag(observed_info_mean)))
            pivot = 2 * np.minimum(pivot_, 1. - pivot_)

            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

            return coverage, intervals[:, 1] - intervals[:, 0], pivot


def test_multitask_lasso_naive_hetero(ntask=2,
                                      nsamples=500 * np.ones(2),
                                      p=100,
                                      global_sparsity=.8,
                                      task_sparsity=.3,
                                      sigma=1. * np.ones(2),
                                      signal_fac=np.array([1., 5.]),
                                      rhos=0. * np.ones(2),
                                      weight=2.):
    nsamples = nsamples.astype(int)
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    while True:
        response_vars, predictor_vars, beta = gaussian_multitask_instance(ntask,
                                                                          nsamples,
                                                                          p,
                                                                          global_sparsity,
                                                                          task_sparsity,
                                                                          sigma,
                                                                          signal,
                                                                          rhos,
                                                                          random_signs=True,
                                                                          equicorrelated=False)[:3]

        feature_weight = weight * np.ones(p)

        sigmas_ = sigma

        perturbations = np.zeros((p, ntask))

        multi_lasso = multi_task_lasso.gaussian(predictor_vars,
                                                response_vars,
                                                feature_weight,
                                                ridge_term=None,
                                                randomizer_scales=1. * sigmas_,
                                                perturbations=perturbations)
        active_signs = multi_lasso.fit()

        dispersions = sigma ** 2

        coverage = []
        pivot = []

        if (active_signs != 0).sum() > 0:

            for i in range(ntask):
                X, y = multi_lasso.loglikes[i].data
                beta_target = np.linalg.pinv(X[:, (active_signs[:, i] != 0)]).dot(X.dot(beta[:, i]))
                Qfeat = np.linalg.inv(X[:, (active_signs[:, i] != 0)].T.dot(X[:, (active_signs[:, i] != 0)]))
                observed_target = np.linalg.pinv(X[:, (active_signs[:, i] != 0)]).dot(y)
                cov_target = Qfeat * dispersions[i]
                alpha = 1. - 0.90
                quantile = ndist.ppf(1 - alpha / 2.)
                intervals = np.vstack([observed_target - quantile * np.sqrt(np.diag(cov_target)),
                                       observed_target + quantile * np.sqrt(np.diag(cov_target))]).T
                coverage.extend((beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1]))
                pivot_ = ndist.cdf((observed_target - beta_target) / np.sqrt(np.diag(cov_target)))
                pivot.extend(2 * np.minimum(pivot_, 1. - pivot_))

            return np.asarray(coverage), intervals[:, 1] - intervals[:, 0], np.asarray(pivot)



def test_coverage(weight,nsim=100):
    cov = []
    len = []
    pivots = []

    cov_naive = []
    len_naive = []
    pivots_naive = []

    ntask = 5

    #penalty = cross_validate_posi_global(ntask=ntask,
                            #nsamples=1000 * np.ones(ntask),
                            #p=50,
                            #global_sparsity=0.95,
                            #task_sparsity=0,
                            #sigma=1. * np.ones(ntask),
                            #signal_fac=0.7,
                            #rhos=0.1 * np.ones(ntask))


    penalty_hetero = cross_validate_posi_hetero(ntask=ntask,
                                         nsamples=1000 * np.ones(ntask),
                                         p=50,
                                         global_sparsity=0.95,
                                         task_sparsity=.2,
                                         sigma=1. * np.ones(ntask),
                                         signal_fac=np.array([1.0, 3.0]),
                                         rhos=.7 * np.ones(ntask),
                                         randomizer_scale = weight)


    penalty_hetero_naive = cross_validate_naive_hetero(ntask=ntask,
                                                nsamples=1000 * np.ones(ntask),
                                                p=50,
                                                global_sparsity=0.95,
                                                task_sparsity=.2,
                                                sigma=1. * np.ones(ntask),
                                                signal_fac=np.array([1.0, 3.0]),
                                                rhos=.7 * np.ones(ntask),
                                                randomizer_scale=weight)

    for n in range(nsim):

        try:

            #coverage, length, pivot = test_multitask_lasso_global(ntask=ntask,
                                                              #nsamples=1000 * np.ones(ntask),
                                                              #p=50,
                                                              #global_sparsity=0.95,
                                                              #task_sparsity=0,
                                                              #sigma=1. * np.ones(ntask),
                                                              #signal_fac=0.7,
                                                              #rhos=0.1 * np.ones(ntask),
                                                              #weight=np.float(penalty))


            # coverage, length, pivot = test_multitask_lasso_naive_global(ntask=ntask,
            #                                                             nsamples=1000 * np.ones(ntask),
            #                                                             p=50,
            #                                                             global_sparsity=0.95,
            #                                                             task_sparsity=0.20,
            #                                                             sigma=1. * np.ones(ntask),
            #                                                             signal_fac=1.,
            #                                                             rhos=0.50 * np.ones(ntask),
            #                                                             weight=1.)

            coverage, length, pivot = test_multitask_lasso_hetero(ntask=ntask,
                                                                  nsamples=1000 * np.ones(ntask),
                                                                  p=50,
                                                                  global_sparsity=0.95,
                                                                  task_sparsity=0.2,
                                                                  sigma=1. * np.ones(ntask),
                                                                  signal_fac=np.array([1.0, 3.0]),
                                                                  rhos=.7 * np.ones(ntask),
                                                                  weight=np.float(penalty_hetero),
                                                                  randomizer_scale = weight)


            coverage_naive, length_naive, pivot_naive = test_multitask_lasso_naive_hetero(ntask=ntask,
                                                                         nsamples=1000 * np.ones(ntask),
                                                                         p=50,
                                                                         global_sparsity=0.95,
                                                                         task_sparsity=0.20,
                                                                         sigma=1. * np.ones(ntask),
                                                                         signal_fac=np.array([1.0, 3.0]),
                                                                         rhos=.7 * np.ones(ntask),
                                                                         weight=np.float(penalty_hetero_naive))

            cov.extend(coverage)
            len.extend(length)
            pivots.extend(pivot)

            cov_naive.extend(coverage_naive)
            len_naive.extend(length_naive)
            pivots_naive.extend(pivot_naive)

            print("iteration completed ", n)
            print("coverage so far ", np.mean(np.asarray(cov_naive)))
            print("length so far ", np.mean(np.asarray(len_naive)))

        except:
            pass

    plt.clf()
    grid = np.linspace(0, 1, 101)
    points = [np.min(np.searchsorted(np.sort(np.asarray(pivots)),i))/np.float(np.shape(pivots)[0]) for i in np.linspace(0, 1, 101)]
    points_naive = [np.min(np.searchsorted(np.sort(np.asarray(pivots_naive)), i)) / np.float(np.shape(pivots_naive)[0]) for i in
              np.linspace(0, 1, 101)]
    plt.plot(grid, points, c='blue', marker='^')
    plt.plot(grid, points_naive, c='red', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.savefig("pivot.png")

    return(np.mean(np.asarray(cov)),np.std(np.asarray(cov)),np.mean(np.asarray(len)),np.std(np.asarray(len)))

def main():

    scale = [1]
    coverage = []
    coverage_er = []
    lengths = []
    lengths_er = []

    for i in range(len(scale)):
        print(scale[i], 'signal')
        results = test_coverage(scale[i], nsim=50)
        coverage = np.append(coverage, results[0])
        print(coverage,"cov")
        coverage_er = np.append(coverage_er, 1.64 * results[1] / np.sqrt(100))
        lengths = np.append(lengths, results[1])
        lengths_er = np.append(lengths_er, 1.64 * results[3] / np.sqrt(100))

    fig = plt.figure()
    plt.errorbar(scale, coverage, yerr=coverage_er)
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.ylabel('Mean Coverage')
    plt.xlabel('Randomizer Scale (sigma_omega)')
    plt.title('Coverage by Randomizer Scale')
    plt.savefig("coverage.png")
    plt.show()

    fig = plt.figure()
    plt.errorbar(scale, lengths, yerr=lengths_er)
    plt.ylabel('Mean Length')
    plt.xlabel('Randomizer Scale (sigma_omega)')
    plt.title('Length by Randomizer Scale')
    plt.savefig("length.png")
    plt.show()


if __name__ == "__main__":
    main()