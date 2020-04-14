from __future__ import division, print_function

import numpy as np
from selection.randomized.lasso import lasso, carved_lasso, selected_targets, full_targets, debiased_targets
from selection.tests.instance import gaussian_instance, exp_instance, normexp_instance, mixednormal_instance, laplace_instance
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import matplotlib.pyplot as plt
from selection.approx_ci.approx_reference import approx_reference, approx_density, \
    approx_reference_adaptive, approx_adaptive_density
from statsmodels.distributions.empirical_distribution import ECDF

def test_approx_pivot(n= 500,
                      p= 100,
                      signal_fac= 1.,
                      s= 5,
                      sigma= 1.,
                      rho= 0.40,
                      randomizer_scale= 1.):

    inst = gaussian_instance
    signal = np.sqrt(signal_fac * 2. * np.log(p))

    while True:
        X, y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=False,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        idx = np.arange(p)
        sigmaX = rho ** np.abs(np.subtract.outer(idx, idx))
        print("snr", beta.T.dot(sigmaX).dot(beta) / ((sigma ** 2.) * n))

        n, p = X.shape

        if n>p:
            dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
            sigma_ = np.sqrt(dispersion)
        else:
            dispersion = None
            sigma_ = np.std(y)

        print("sigma estimated and true ", sigma, sigma_)

        W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

        conv = lasso.gaussian(X,
                              y,
                              W,
                              randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0

        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = selected_targets(conv.loglike,
                                          conv._W,
                                          nonzero,
                                          dispersion=dispersion)

        grid_num = 501
        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
        pivot = []
        for m in range(nonzero.sum()):
            observed_target_uni = (observed_target[m]).reshape((1,))
            cov_target_uni = (np.diag(cov_target)[m]).reshape((1,1))
            cov_target_score_uni = cov_target_score[m,:].reshape((1, p))
            mean_parameter = beta_target[m]
            grid = np.linspace(- 25., 25., num=grid_num)
            grid_indx_obs = np.argmin(np.abs(grid - observed_target_uni))

            approx_log_ref= approx_reference(grid,
                                             observed_target_uni,
                                             cov_target_uni,
                                             cov_target_score_uni,
                                             conv.observed_opt_state,
                                             conv.cond_mean,
                                             conv.cond_cov,
                                             conv.logdens_linear,
                                             conv.A_scaling,
                                             conv.b_scaling)

            area_cum = approx_density(grid,
                                      mean_parameter,
                                      cov_target_uni,
                                      approx_log_ref)

            print("check ", area_cum)

            pivot.append(1. - area_cum[grid_indx_obs])
            print("variable completed ", m+1)
        return pivot

def test_approx_pivot_carved(n= 100,
                             p= 50,
                             signal_fac= 1.,
                             s= 5,
                             sigma= 1.,
                             rho= 0.40,
                             split_proportion=0.50):

    inst = laplace_instance
    signal = np.sqrt(signal_fac * 2. * np.log(p))

    while True:
        X, y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=False,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        idx = np.arange(p)
        sigmaX = rho ** np.abs(np.subtract.outer(idx, idx))
        print("snr", beta.T.dot(sigmaX).dot(beta) / ((sigma ** 2.) * n))
        n, p = X.shape

        if n>p:
            dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
            sigma_ = np.sqrt(dispersion)
        else:
            dispersion = None
            sigma_ = np.std(y)

        print("sigma estimated and true ", sigma, sigma_)
        randomization_cov = ((sigma_ ** 2) * ((1. - split_proportion) / split_proportion)) * sigmaX
        lam_theory = sigma_ * 1. * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))

        conv = carved_lasso.gaussian(X,
                                     y,
                                     noise_variance=sigma_ ** 2.,
                                     rand_covariance="True",
                                     randomization_cov=randomization_cov / float(n),
                                     feature_weights=np.ones(X.shape[1]) * lam_theory,
                                     subsample_frac=split_proportion)

        signs = conv.fit()
        nonzero = signs != 0

        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = selected_targets(conv.loglike,
                                          conv._W,
                                          nonzero,
                                          dispersion=dispersion)

        grid_num = 501
        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
        pivot = []
        for m in range(nonzero.sum()):
            observed_target_uni = (observed_target[m]).reshape((1,))
            cov_target_uni = (np.diag(cov_target)[m]).reshape((1, 1))
            cov_target_score_uni = cov_target_score[m, :].reshape((1, p))
            mean_parameter = beta_target[m]
            grid = np.linspace(- 25., 25., num=grid_num)
            grid_indx_obs = np.argmin(np.abs(grid - observed_target_uni))

            approx_log_ref = approx_reference(grid,
                                              observed_target_uni,
                                              cov_target_uni,
                                              cov_target_score_uni,
                                              conv.observed_opt_state,
                                              conv.cond_mean,
                                              conv.cond_cov,
                                              conv.logdens_linear,
                                              conv.A_scaling,
                                              conv.b_scaling)

            area_cum = approx_density(grid,
                                      mean_parameter,
                                      cov_target_uni,
                                      approx_log_ref)


            pivot.append(1. - area_cum[grid_indx_obs])
            print("variable completed ", m + 1)
        return pivot

def EDCF_pivot(nsim=300):
    _pivot=[]
    for i in range(nsim):
        _pivot.extend(test_approx_pivot(n= 300,
                                        p= 50,
                                        signal_fac= 0.25,
                                        s= 5,
                                        sigma= 1.,
                                        rho= 0.40,
                                        randomizer_scale= 1.))
        print("iteration completed ", i)
    plt.clf()
    ecdf_MLE = ECDF(np.asarray(_pivot))
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, ecdf_MLE(grid), c='blue', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.show()

#EDCF_pivot(nsim=300)

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

def plotPivot(pivot):
    robjects.r("""
    
               pivot_plot <- function(pivot, outpath='/Users/psnigdha/Research/Pivot_selective_MLE/ArXiV-2/submission-revision/', resolution=350, height=10, width=10)
               {
                    pivot = as.vector(pivot)
                    outfile = paste(outpath, 'pivot_LASSO_n200_gaussian_snr20.png', sep="")
                    png(outfile, res = resolution, width = width, height = height, units = 'cm')
                    par(mar=c(5,4,2,2)+0.1)
                    plot(ecdf(pivot), lwd=8, lty = 2, col="#000080", main="Model-4", ylab="", xlab="", cex.main=0.95)
                    abline(a = 0, b = 1, lwd=5, col="black")
                    dev.off()
               }                       
               """)

    R_plot = robjects.globalenv['pivot_plot']
    r_pivot = robjects.r.matrix(pivot, nrow=pivot.shape[0], ncol=1)
    R_plot(r_pivot)


def plotPivot_randomization(pivot_carved, pivot_randomized):
    robjects.r("""

               pivot_plot <- function(pivot_carved, pivot_randomized, 
               outpath='/Users/psnigdha/Research/Pivot_selective_MLE/ArXiV-2/submission-revision/', resolution=350, height=10, width=10)
               {
                    pivot_carved = as.vector(pivot_carved)
                    pivot_randomized = as.vector(pivot_randomized)
                    outfile = paste(outpath, 'randomized_pivot_LASSO_n100p1000_laplace_snr15.png', sep="")
                    png(outfile, res = resolution, width = width, height = height, units = 'cm')
                    par(mar=c(5,4,2,2)+0.1)
                    plot(ecdf(pivot_randomized), lwd=8, lty = 2, col="#000080", main="Model-4", ylab="", xlab="", cex.main=0.95)
                    plot(ecdf(pivot_carved), lwd=3, verticals=TRUE, add=TRUE, col='darkred')                    
                    abline(a = 0, b = 1, lwd=5, col="black")
                    dev.off()
               }                       
               """)

    R_plot = robjects.globalenv['pivot_plot']
    r_pivot_carved = robjects.r.matrix(pivot_carved, nrow=pivot_carved.shape[0], ncol=1)
    r_pivot_randomized = robjects.r.matrix(pivot_randomized, nrow=pivot_randomized.shape[0], ncol=1)
    R_plot(r_pivot_carved, r_pivot_randomized)

def main(nsim=200):
    _pivot=[]
    for i in range(nsim):
        _pivot.extend(test_approx_pivot_carved(n= 300,
                                               p= 500,
                                               signal_fac= 1.,
                                               s= 10,
                                               sigma= 1.,
                                               rho= 0.20,
                                               split_proportion=0.50))
        print("iteration completed ", i)

    plotPivot(np.asarray(_pivot))

#main()

def compare_pivots_highD(nsim=200):
    _carved_pivot = []
    _randomized_pivot = []

    for i in range(nsim):
        _carved_pivot.extend(test_approx_pivot_carved(n= 100,
                                                      p= 1000,
                                                      signal_fac= 0.6,
                                                      s= 10,
                                                      sigma= 1.,
                                                      rho= 0.40,
                                                      split_proportion=0.50))

        _randomized_pivot.extend(test_approx_pivot(n= 100,
                                                   p= 1000,
                                                   signal_fac= 0.6,
                                                   s= 10,
                                                   sigma= 1.,
                                                   rho= 0.40,
                                                   randomizer_scale= 1.))

        print("iteration completed ", i)

    plotPivot_randomization(np.asarray(_carved_pivot), np.asarray(_randomized_pivot))

#compare_pivots_highD(nsim=250)

