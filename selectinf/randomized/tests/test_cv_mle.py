from __future__ import division

import numpy as np, os, itertools
import pandas as pd

import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri
rpy.numpy2ri.activate()
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from scipy.stats import norm as ndist
from selectinf.randomized.lasso import lasso, full_targets, selected_targets, debiased_targets
from selectinf.algorithms.lasso import ROSI

def sim_xy(n, p, nval, rho=0, s=5, beta_type=2, snr=1):

    rpy.r('''

    #' Predictors and responses generation.
    #'
    #' Generate a predictor matrix x, and response vector y, following a specified
    #'   setup.  Actually, two pairs of predictors and responses are generated:
    #'   one for training, and one for validation.
    #'
    #' @param n,p The number of training observations, and the number of predictors.
    #' @param nval The number of validation observations.
    #' @param rho Parameter that drives pairwise correlations of the predictor
    #'   variables; specifically, predictors i and j have population correlation
    #'   rho^abs(i-j). Default is 0.
    #' @param s number of nonzero coefficients in the underlying regression model.
    #'   Default is 5. (Ignored if beta.type is 4, in which case the number of
    #'   nonzero coefficients is 6; and if beta.type is 5, it is interpreted as a
    #'   the number of strongly nonzero coefficients in a weak sparsity model.)
    #' @param beta.type Integer taking values in between 1 and 5, used to specify
    #'   the pattern of nonzero coefficients in the underlying regression model; see
    #'   details below. Default is 1.
    #' @param snr Desired signal-to-noise ratio (SNR), i.e., var(mu)/sigma^2 where
    #'   mu is mean and sigma^2 is the error variance. The error variance is set so
    #'   that the given SNR is achieved. Default is 1.
    #' @return A list with the following components: x, y, xval, yval, Sigma, beta,
    #'   and sigma.
    #'
    #' @details The data model is: \eqn{Y \sim N(X\beta, \sigma^2 I)}.
    #'   The predictor variables have covariance matrix Sigma, with (i,j)th entry
    #'   rho^abs(i-j). The error variance sigma^2 is set according to the desired
    #'   signal-to-noise ratio. The first 4 options for the nonzero pattern
    #'   of the underlying regression coefficients beta follow the simulation setup
    #'   in Bertsimas, King, and Mazumder (2016), and the 5th is a weak sparsity
    #'   option:
    #'   \itemize{
    #'   \item 1: beta has s components of 1, occurring at (roughly) equally-spaced
    #'      indices in between 1 and p
    #'   \item 2: beta has its first s components equal to 1
    #'   \item 3: beta has its first s components taking nonzero values, where the
    #'       decay in a linear fashion from 10 to 0.5
    #'   \item 4: beta has its first 6 components taking the nonzero values -10,-6,
    #'       -2,2,6,10
    #'   \item 5: beta has its first s components equal to 1, and the rest decaying
    #'       to zero at an exponential rate
    #'   }
    #'
    #' @author Trevor Hastie, Rob Tibshirani, Ryan Tibshirani
    #' @references Simulation setup based on "Best subset selection via a modern
    #'   optimization lens" by Dimitris Bertsimas, Angela King, and Rahul Mazumder,
    #'   Annals of Statistics, 44(2), 813-852, 2016.
    #' @example examples/ex.fs.R
    #' @export sim.xy

    sim.xy = function(n, p, nval, rho=0, s=5, beta.type=1, snr=1) {
      # Generate predictors
      x = matrix(rnorm(n*p),n,p)
      xval = matrix(rnorm(nval*p),nval,p)

      # Introduce autocorrelation, if needed
      if (rho != 0) {
        inds = 1:p
        Sigma = rho^abs(outer(inds, inds, "-"))
        obj = svd(Sigma)
        Sigma.half = obj$u %*% (sqrt(diag(obj$d))) %*% t(obj$v)
        x = x %*% Sigma.half
        xval = xval %*% Sigma.half
      }
      else Sigma = diag(1,p)

      # Generate underlying coefficients
      s = min(s,p)
      beta = rep(0,p)
      if (beta.type==1) {
        beta[round(seq(1,p,length=s))] = 1
      } else if (beta.type==2) {
        beta[1:s] = 1
      } else if (beta.type==3) {
        beta[1:s] = seq(10,0.5,length=s)
      } else if (beta.type==4) {
        beta[1:6] = c(-10,-6,-2,2,6,10)
      } else {
        beta[1:s] = 1
        beta[(s+1):p] = 0.5^(1:(p-s))
      }

      # Set snr based on sample variance on infinitely large test set
      vmu = as.numeric(t(beta) %*% Sigma %*% beta)
      sigma = sqrt(vmu/snr)

      # Generate responses
      y = as.numeric(x %*% beta + rnorm(n)*sigma)
      yval = as.numeric(xval %*% beta + rnorm(nval)*sigma)

      list(x=x,y=y,xval=xval,yval=yval,Sigma=Sigma,beta=beta,sigma=sigma)
    }

    sim_xy = sim.xy
    ''')

    r_simulate = rpy.globalenv['sim_xy']
    sim = r_simulate(n, p, nval, rho, s, beta_type, snr)
    X = np.array(sim.rx2('x'))
    y = np.array(sim.rx2('y'))
    X_val = np.array(sim.rx2('xval'))
    y_val = np.array(sim.rx2('yval'))
    Sigma = np.array(sim.rx2('Sigma'))
    beta = np.array(sim.rx2('beta'))
    sigma = np.array(sim.rx2('sigma'))

    return X, y, X_val, y_val, Sigma, beta, sigma


def selInf_R(X, y, beta, lam, sigma, Type, alpha=0.1):
    rpy.r('''
               library("selectiveInference")
               selInf = function(X, y, beta, lam, sigma, Type, alpha= 0.1){
               y = as.matrix(y)
               X = as.matrix(X)
               beta = as.matrix(beta)
               lam = as.matrix(lam)[1,1]
               sigma = as.matrix(sigma)[1,1]
               Type = as.matrix(Type)[1,1]
               if(Type == 1){
                   type = "full"} else{
                   type = "partial"}
               inf = fixedLassoInf(x = X, y = y, beta = beta, lambda=lam, family = "gaussian",
                                   intercept=FALSE, sigma=sigma, alpha=alpha, type=type)
               return(list(ci = inf$ci, pvalue = inf$pv))}
               ''')

    inf_R = rpy.globalenv['selInf']
    n, p = X.shape
    r_X = rpy.r.matrix(X, nrow=n, ncol=p)
    r_y = rpy.r.matrix(y, nrow=n, ncol=1)
    r_beta = rpy.r.matrix(beta, nrow=p, ncol=1)
    r_lam = rpy.r.matrix(lam, nrow=1, ncol=1)
    r_sigma = rpy.r.matrix(sigma, nrow=1, ncol=1)
    r_Type = rpy.r.matrix(Type, nrow=1, ncol=1)
    output = inf_R(r_X, r_y, r_beta, r_lam, r_sigma, r_Type)
    ci = np.array(output.rx2('ci'))
    pvalue = np.array(output.rx2('pvalue'))
    return ci, pvalue


def glmnet_lasso(X, y, lambda_val):
    rpy.r('''
                library(glmnet)
                glmnet_LASSO = function(X,y, lambda){
                y = as.matrix(y)
                X = as.matrix(X)
                lam = as.matrix(lambda)[1,1]
                n = nrow(X)

                fit = glmnet(X, y, standardize=FALSE, intercept=FALSE, thresh=1.e-10)
                estimate = coef(fit, s=lam, exact=TRUE, x=X, y=y)[-1]
                fit.cv = cv.glmnet(X, y, standardize=FALSE, intercept=FALSE, thresh=1.e-10)
                estimate.1se = coef(fit, s=fit.cv$lambda.1se, exact=TRUE, x=X, y=y)[-1]
                estimate.min = coef(fit, s=fit.cv$lambda.min, exact=TRUE, x=X, y=y)[-1]
                return(list(estimate = estimate, estimate.1se = estimate.1se, estimate.min = estimate.min, lam.min = fit.cv$lambda.min, lam.1se = fit.cv$lambda.1se))
                }''')

    lambda_R = rpy.globalenv['glmnet_LASSO']
    n, p = X.shape
    r_X = rpy.r.matrix(X, nrow=n, ncol=p)
    r_y = rpy.r.matrix(y, nrow=n, ncol=1)
    r_lam = rpy.r.matrix(lambda_val, nrow=1, ncol=1)

    val = lambda_R(r_X, r_y, r_lam)
    estimate = np.array(val.rx2('estimate'))
    estimate_1se = np.array(val.rx2('estimate.1se'))
    estimate_min = np.array(val.rx2('estimate.min'))
    lam_min = np.asscalar(np.array(val.rx2('lam.min')))
    lam_1se = np.asscalar(np.array(val.rx2('lam.1se')))
    return estimate, estimate_1se, estimate_min, lam_min, lam_1se


def coverage(intervals, pval, target, truth):
    pval_alt = (pval[truth != 0]) < 0.1
    if pval_alt.sum() > 0:
        avg_power = np.mean(pval_alt)
    else:
        avg_power = 0.
    return np.mean((target > intervals[:, 0]) * (target < intervals[:, 1])), avg_power


def BHfilter(pval, q=0.2):
    rpy.r.assign('pval', pval)
    rpy.r.assign('q', q)
    rpy.r('Pval = p.adjust(pval, method="BH")')
    rpy.r('S = which((Pval < q)) - 1')
    S = rpy.r('S')
    ind = np.zeros(pval.shape[0], np.bool)
    ind[np.asarray(S, np.int)] = 1
    return ind


def relative_risk(est, truth, Sigma):
    if (truth != 0).sum() > 0:
        return (est - truth).T.dot(Sigma).dot(est - truth) / truth.T.dot(Sigma).dot(truth)
    else:
        return (est - truth).T.dot(Sigma).dot(est - truth)

from rpy2 import robjects

def plotRisk(df_risk):
    robjects.r("""
               library("ggplot2")
               library("magrittr")
               library("tidyr")
               library("dplyr")

               plot_risk <- function(df_risk, outpath="plots/", resolution=300, height= 7.5, width=15)
                { 
                   date = 1:length(unique(df_risk$snr))
                   df_risk = filter(df_risk, metric == "Full")
                   df = cbind(df_risk, date)
                   risk = df %>%
                   gather(key, value, sel.MLE, rand.LASSO, LASSO) %>%
                   ggplot(aes(x=date, y=value, colour=key, shape=key, linetype=key)) +
                   geom_point(size=3) +
                   geom_line(aes(linetype=key), size=1) +
                   ylim(0.01,1.2)+
                   labs(y="relative risk", x = "Signal regimes: snr") +
                   scale_x_continuous(breaks=1:length(unique(df_risk$snr)), label = sapply(df_risk$snr, toString)) +
                   theme(legend.position="top", legend.title = element_blank())
                   indices = sort(c("sel.MLE", "rand.LASSO", "LASSO"), index.return= TRUE)$ix
                   names = c("sel-MLE", "rand-LASSO", "LASSO")
                   risk = risk + scale_color_manual(labels = names[indices], values=c("#008B8B", "#104E8B","#B22222")[indices]) +
                   scale_shape_manual(labels = names[indices], values=c(15, 17, 16)[indices]) +
                                      scale_linetype_manual(labels = names[indices], values = c(1,1,2)[indices])
                                      outfile = paste(outpath, 'risk.png', sep="")
                   outfile = paste(outpath, 'risk.png', sep="")                   
                   ggsave(outfile, plot = risk, dpi=resolution, dev='png', height=height, width=width, units="cm")}
                """)

    #pandas2ri.activate()
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_df_risk = robjects.conversion.py2rpy(df_risk)
    R_plot = robjects.globalenv['plot_risk']
    R_plot(r_df_risk)


def plotCoveragePower(df_inference):
    robjects.r("""
               library("ggplot2")
               library("magrittr")
               library("tidyr")
               library("reshape")
               library("cowplot")
               library("dplyr")

               plot_coverage_lengths <- function(df_inference, outpath="plots/", 
                                                 resolution=200, height_plot1= 6.5, width_plot1=12, 
                                                 height_plot2=13, width_plot2=13)
               {
                 snr.len = length(unique(df_inference$snr))
                 df_inference = arrange(df_inference, method)
                 target = toString(df_inference$target[1])
                 df = data.frame(snr = sapply(unique(df_inference$snr), toString),
                                 MLE = 100*df_inference$coverage[((2*snr.len)+1):(3*snr.len)],
                                 Lee = 100*df_inference$coverage[1:snr.len],
                                 Naive = 100*df_inference$coverage[((3*snr.len)+1):(4*snr.len)])
                 if(target== "selected"){
                      data.m <- melt(df, id.vars='snr')
                      coverage = ggplot(data.m, aes(snr, value)) + 
                                 geom_bar(aes(fill = variable), width = 0.4, position = position_dodge(width=0.5), stat="identity") + 
                                 geom_hline(yintercept = 90, linetype="dotted") +
                                 labs(y="coverage: partial", x = "Signal regimes: snr") +
                                 theme(legend.position="top", 
                                       legend.title = element_blank()) 
                      coverage = coverage + 
                                 scale_fill_manual(labels = c("MLE-based","Lee", "Naive"), values=c("#008B8B", "#B22222", "#FF6347"))} else{
                 df = cbind(df, Liu = 100*df_inference$coverage[((snr.len)+1):(2*snr.len)])
                 df <- df[c("snr", "MLE", "Liu", "Lee", "Naive")]
                 data.m <- melt(df, id.vars='snr')
                 coverage = ggplot(data.m, aes(snr, value)) + 
                            geom_bar(aes(fill = variable), width = 0.4, position = position_dodge(width=0.5), stat="identity") + 
                            geom_hline(yintercept = 90, linetype="dotted") +
                            labs(y="coverage: full", x = "Signal regimes: snr") +
                            theme(legend.position="top", legend.title = element_blank()) 
                  coverage = coverage + 
                  scale_fill_manual(labels = c("MLE-based", "Liu", "Lee", "Naive"), values=c("#008B8B", "#104E8B", "#B22222", "#FF6347"))}

                 outfile = paste(outpath, 'coverage.png', sep="")
                 ggsave(outfile, plot = coverage, dpi=resolution, dev='png', height=height_plot1, width=width_plot1, units="cm")

                 df = data.frame(snr = sapply(unique(df_inference$snr), toString),
                                 MLE = 100*df_inference$sel.power[((2*snr.len)+1):(3*snr.len)],
                                 Lee = 100*df_inference$sel.power[1:snr.len])
                 if(target== "selected"){
                   data.m <- melt(df, id.vars='snr')
                   sel_power = ggplot(data.m, aes(snr, value)) + 
                               geom_bar(aes(fill = variable), width = 0.4, position = position_dodge(width=0.5), stat="identity") + 
                               labs(y="power: partial", x = "Signal regimes: snr") +
                               theme(legend.position="top", legend.title = element_blank()) 
                   sel_power = sel_power + scale_fill_manual(labels = c("MLE-based","Lee"), values=c("#008B8B", "#B22222"))} else{
                   df = cbind(df, Liu = 100*df_inference$sel.power[((snr.len)+1):(2*snr.len)])
                   df <- df[,c("snr", "MLE", "Liu", "Lee")]
                   data.m <- melt(df, id.vars='snr')
                   sel_power = ggplot(data.m, aes(snr, value)) + 
                               geom_bar(aes(fill = variable), width = 0.4, position = position_dodge(width=0.5), stat="identity") + 
                               labs(y="power: full", x = "Signal regimes: snr") +
                               theme(legend.position="top", legend.title = element_blank()) 
                   sel_power = sel_power + scale_fill_manual(labels = c("MLE-based","Liu","Lee"), values=c("#008B8B", "#104E8B", "#B22222"))}

                 outfile = paste(outpath, 'selective_power.png', sep="")
                 ggsave(outfile, plot = sel_power, dpi=resolution, dev='png', height=height_plot1, width=width_plot1, units="cm")

               if(target== "selected"){
                   test_data <-data.frame(MLE = filter(df_inference, method == "MLE")$length,
                   Lee = filter(df_inference, method == "Lee")$length,
                   Naive = filter(df_inference, method == "Naive")$length,
                   date = 1:length(unique(df_inference$snr)))
                   lengths = test_data %>%
                             gather(key, value, MLE, Lee, Naive) %>%
                             ggplot(aes(x=date, y=value, colour=key, shape=key, linetype=key)) +
                             geom_point(size=3) +
                             geom_line(aes(linetype=key), size=1) +
                             ylim(0.,max(test_data$MLE, test_data$Lee, test_data$Naive) + 0.2)+
                             labs(y="lengths:partial", x = "Signal regimes: snr") +
                             scale_x_continuous(breaks=1:length(unique(df_inference$snr)), label = sapply(unique(df_inference$snr), toString))+
                             theme(legend.position="top", legend.title = element_blank())

                   indices = sort(c("MLE", "Lee", "Naive"), index.return= TRUE)$ix
                   names = c("MLE-based", "Lee", "Naive")
                   lengths = lengths + scale_color_manual(labels = names[indices], values=c("#008B8B","#B22222", "#FF6347")[indices]) +
                             scale_shape_manual(labels = names[indices], values=c(15, 17, 16)[indices]) +
                             scale_linetype_manual(labels = names[indices], values = c(1,1,2)[indices])} else{
                   test_data <-data.frame(MLE = filter(df_inference, method == "MLE")$length,
                                          Lee = filter(df_inference, method == "Lee")$length,
                                          Naive = filter(df_inference, method == "Naive")$length,
                                          Liu = filter(df_inference, method == "Liu")$length,
                                          date = 1:length(unique(df_inference$snr)))
                   lengths= test_data %>%
                            gather(key, value, MLE, Lee, Naive, Liu) %>%
                            ggplot(aes(x=date, y=value, colour=key, shape=key, linetype=key)) +
                            geom_point(size=3) +
                            geom_line(aes(linetype=key), size=1) +
                            ylim(0.,max(test_data$MLE, test_data$Lee, test_data$Naive, test_data$Liu) + 0.2)+
                            labs(y="lengths: full", x = "Signal regimes: snr") +
                            scale_x_continuous(breaks=1:length(unique(df_inference$snr)), label = sapply(unique(df_inference$snr), toString))+
                            theme(legend.position="top", legend.title = element_blank())

                   indices = sort(c("MLE", "Liu", "Lee", "Naive"), index.return= TRUE)$ix
                   names = c("MLE-based", "Lee", "Naive", "Liu")
                   lengths = lengths + scale_color_manual(labels = names[indices], values=c("#008B8B","#B22222", "#FF6347", "#104E8B")[indices]) +
                             scale_shape_manual(labels = names[indices], values=c(15, 17, 16, 15)[indices]) +
                             scale_linetype_manual(labels = names[indices], values = c(1,1,2,1)[indices])}

               prop = filter(df_inference, method == "Lee")$prop.infty
               df = data.frame(snr = sapply(unique(df_inference$snr), toString),
               infinite = 100*prop)
               data.prop <- melt(df, id.vars='snr')
               pL = ggplot(data.prop, aes(snr, value)) +
                    geom_bar(aes(fill = variable), width = 0.4, position = position_dodge(width=0.5), stat="identity") + 
                    labs(y="infinite intervals (%)", x = "Signal regimes: snr") +
                    theme(legend.position="top", 
                    legend.title = element_blank()) 
               pL = pL + scale_fill_manual(labels = c("Lee"), values=c("#B22222"))
               prow <- plot_grid( pL + theme(legend.position="none"),
                                  lengths + theme(legend.position="none"),
                                  align = 'vh',
                                  hjust = -1,
                                  ncol = 1)

               legend <- get_legend(lengths+ theme(legend.direction = "horizontal",legend.justification="center" ,legend.box.just = "bottom"))
               p <- plot_grid(prow, ncol=1, legend, rel_heights = c(2., .2)) 
               outfile = paste(outpath, 'length.png', sep="")
               ggsave(outfile, plot = p, dpi=resolution, dev='png', height=height_plot2, width=width_plot2, units="cm")}
               """)

    #pandas2ri.activate()
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_df_inference = robjects.conversion.py2rpy(df_inference)
    R_plot = robjects.globalenv['plot_coverage_lengths']
    R_plot(r_df_inference)

def comparison_cvmetrics_selected(n=500,
                                  p=100,
                                  nval=500,
                                  rho=0.35,
                                  s=5,
                                  beta_type=1,
                                  snr=0.20,
                                  randomizer_scale=np.sqrt(0.50),
                                  full_dispersion=True,
                                  tuning_nonrand="lambda.min",
                                  tuning_rand="lambda.1se"):

    (X,
     y,
     _,
     _,
     Sigma,
     beta,
     sigma) = sim_xy(n=n,
                     p=p,
                     nval=nval,
                     rho=rho,
                     s=s,
                     beta_type=beta_type,
                     snr=snr)
    true_mean = X.dot(beta)

    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1)))
    y = y - y.mean()
    true_set = np.asarray([u for u in range(p) if beta[u] != 0])

    if full_dispersion:
        dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
        sigma_ = np.sqrt(dispersion)
    else:
        dispersion = None
        sigma_ = np.std(y)
    print("estimated and true sigma", sigma, sigma_)

    lam_theory = sigma_ * 1. * np.mean(np.fabs(np.dot(X.T,
                                                      np.random.standard_normal((n,
                                                                                 2000)))).max(0))
    (glm_LASSO_theory,
     glm_LASSO_1se,
     glm_LASSO_min,
     lam_min,
     lam_1se) = glmnet_lasso(X,
                             y,
                             lam_theory/float(n))

    if tuning_nonrand == "lambda.min":
        lam_LASSO = lam_min
        glm_LASSO = glm_LASSO_min
    elif tuning_nonrand == "lambda.1se":
        lam_LASSO = lam_1se
        glm_LASSO = glm_LASSO_1se
    else:
        lam_LASSO = lam_theory/float(n)
        glm_LASSO = glm_LASSO_theory

    active_LASSO = (glm_LASSO != 0)
    nactive_LASSO = active_LASSO.sum()
    active_set_LASSO = np.asarray([r for r in range(p) if active_LASSO[r]])
    active_LASSO_bool = np.asarray([(np.in1d(active_set_LASSO[z], true_set).sum() > 0) for z in range(nactive_LASSO)], np.bool)

    rel_LASSO = np.zeros(p)
    Lee_nreport = 0
    bias_Lee = 0.
    bias_naive = 0.

    if nactive_LASSO > 0:
        post_LASSO_OLS = np.linalg.pinv(X[:, active_LASSO]).dot(y)
        rel_LASSO[active_LASSO] = post_LASSO_OLS
        Lee_target = np.linalg.pinv(X[:, active_LASSO]).dot(X.dot(beta))
        try:
            Lee_intervals, Lee_pval = selInf_R(X, y, glm_LASSO, n * lam_LASSO, sigma_, Type=0, alpha=0.1)
        except:
            Lee_intervals, Lee_pval = np.array([]), np.array([])
            
        if (Lee_pval.shape[0] == Lee_target.shape[0]):

            cov_Lee, selective_Lee_power = coverage(Lee_intervals, Lee_pval, Lee_target, beta[active_LASSO])
            inf_entries_bool = np.isinf(Lee_intervals[:, 1] - Lee_intervals[:, 0])
            inf_entries = np.mean(inf_entries_bool)
            if inf_entries == 1.:
                length_Lee = 0.
            else:
                length_Lee = np.mean((Lee_intervals[:, 1] - Lee_intervals[:, 0])[~inf_entries_bool])
            power_Lee = ((active_LASSO_bool) * (np.logical_or((0. < Lee_intervals[:, 0]), (0. > Lee_intervals[:, 1])))) \
                            .sum() / float((beta != 0).sum())
            Lee_discoveries = BHfilter(Lee_pval, q=0.1)
            power_Lee_BH = (Lee_discoveries * active_LASSO_bool).sum() / float((beta != 0).sum())
            fdr_Lee_BH = (Lee_discoveries * ~active_LASSO_bool).sum() / float(max(Lee_discoveries.sum(), 1.))
            bias_Lee = np.mean(glm_LASSO[active_LASSO] - Lee_target)

            naive_sd = sigma_ * np.sqrt(np.diag((np.linalg.inv(X[:, active_LASSO].T.dot(X[:, active_LASSO])))))
            naive_intervals = np.vstack([post_LASSO_OLS - 1.65 * naive_sd,
                                         post_LASSO_OLS + 1.65 * naive_sd]).T
            naive_pval = 2 * ndist.cdf(np.abs(post_LASSO_OLS) / naive_sd)
            cov_naive, selective_naive_power = coverage(naive_intervals, naive_pval, Lee_target, beta[active_LASSO])
            length_naive = np.mean(naive_intervals[:, 1] - naive_intervals[:, 0])
            power_naive = ((active_LASSO_bool) * (
                np.logical_or((0. < naive_intervals[:, 0]), (0. > naive_intervals[:, 1])))).sum() / float(
                (beta != 0).sum())
            naive_discoveries = BHfilter(naive_pval, q=0.1)
            power_naive_BH = (naive_discoveries * active_LASSO_bool).sum() / float((beta != 0).sum())
            fdr_naive_BH = (naive_discoveries * ~active_LASSO_bool).sum() / float(max(naive_discoveries.sum(), 1.))
            bias_naive = np.mean(rel_LASSO[active_LASSO] - Lee_target)

            partial_Lasso_risk = (glm_LASSO[active_LASSO]-Lee_target).T.dot(glm_LASSO[active_LASSO]-Lee_target)
            partial_relLasso_risk = (post_LASSO_OLS - Lee_target).T.dot(post_LASSO_OLS - Lee_target)

        else:
            Lee_nreport = 1
            cov_Lee, length_Lee, inf_entries, power_Lee, power_Lee_BH, fdr_Lee_BH, selective_Lee_power = [0., 0., 0., 0., 0., 0., 0.]
            cov_naive, length_naive, power_naive, power_naive_BH, fdr_naive_BH, selective_naive_power = [0., 0., 0., 0., 0., 0.]
            naive_discoveries = np.zeros(1)
            Lee_discoveries = np.zeros(1)
            partial_Lasso_risk,  partial_relLasso_risk = [0., 0.]
    elif nactive_LASSO == 0:
        Lee_nreport = 1
        cov_Lee, length_Lee, inf_entries, power_Lee, power_Lee_BH, fdr_Lee_BH, selective_Lee_power = [0., 0., 0., 0., 0., 0., 0.]
        cov_naive, length_naive, power_naive, power_naive_BH, fdr_naive_BH, selective_naive_power = [0., 0., 0., 0., 0., 0.]
        naive_discoveries = np.zeros(1)
        Lee_discoveries = np.zeros(1)
        partial_Lasso_risk, partial_relLasso_risk = [0., 0.]

    if tuning_rand == "lambda.min":
        randomized_lasso = lasso.gaussian(X,
                                          y,
                                          feature_weights=n * lam_min * np.ones(p),
                                          randomizer_scale= np.sqrt(n) * randomizer_scale * sigma_)
    elif tuning_rand == "lambda.1se":
        randomized_lasso = lasso.gaussian(X,
                                          y,
                                          feature_weights=n * lam_1se * np.ones(p),
                                          randomizer_scale= np.sqrt(n) * randomizer_scale * sigma_)
    else:
        randomized_lasso = lasso.gaussian(X,
                                          y,
                                          feature_weights= lam_theory * np.ones(p),
                                          randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)
    signs = randomized_lasso.fit()
    nonzero = signs != 0
    active_set_rand = np.asarray([t for t in range(p) if nonzero[t]])
    active_rand_bool = np.asarray([(np.in1d(active_set_rand[x], true_set).sum() > 0) for x in range(nonzero.sum())], np.bool)
    sel_MLE = np.zeros(p)
    ind_est = np.zeros(p)
    randomized_lasso_est = np.zeros(p)
    randomized_rel_lasso_est = np.zeros(p)
    MLE_nreport = 0

    if nonzero.sum() > 0:
        target_randomized = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = selected_targets(randomized_lasso.loglike,
                                          randomized_lasso._W,
                                          nonzero,
                                          dispersion=dispersion)

        result = randomized_lasso.selective_MLE(observed_target,
                                                cov_target,
                                                cov_target_score)[0]

        MLE_estimate = result['MLE']
        ind_unbiased_estimator = result['unbiased']

        sel_MLE[nonzero] = MLE_estimate
        ind_est[nonzero] = ind_unbiased_estimator
        MLE_intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])
        MLE_pval = np.asarray(result['pvalue'])

        randomized_lasso_est = randomized_lasso.initial_soln
        randomized_rel_lasso_est = randomized_lasso._beta_full

        cov_MLE, selective_MLE_power = coverage(MLE_intervals, MLE_pval, target_randomized, beta[nonzero])
        length_MLE = np.mean(MLE_intervals[:, 1] - MLE_intervals[:, 0])
        power_MLE = ((active_rand_bool) * (
            np.logical_or((0. < MLE_intervals[:, 0]), (0. > MLE_intervals[:, 1])))).sum() / float((beta != 0).sum())
        MLE_discoveries = BHfilter(MLE_pval, q=0.1)
        power_MLE_BH = (MLE_discoveries * active_rand_bool).sum() / float((beta != 0).sum())
        fdr_MLE_BH = (MLE_discoveries * ~active_rand_bool).sum() / float(max(MLE_discoveries.sum(), 1.))
        bias_MLE = np.mean(MLE_estimate - target_randomized)

        partial_MLE_risk = (MLE_estimate - target_randomized).T.dot(MLE_estimate - target_randomized)
        partial_ind_risk = (ind_unbiased_estimator - target_randomized).T.dot(ind_unbiased_estimator - target_randomized)
        partial_randLasso_risk = (randomized_lasso_est[nonzero] - target_randomized).T.dot(randomized_lasso_est[nonzero] - target_randomized)
        partial_relrandLasso_risk = (randomized_rel_lasso_est[nonzero] - target_randomized).T.dot(randomized_rel_lasso_est[nonzero] - target_randomized)

    else:
        MLE_nreport = 1
        cov_MLE, length_MLE, power_MLE, power_MLE_BH, fdr_MLE_BH, bias_MLE, selective_MLE_power = [0., 0., 0., 0., 0., 0., 0.]
        MLE_discoveries = np.zeros(1)
        partial_MLE_risk, partial_ind_risk, partial_randLasso_risk, partial_relrandLasso_risk = [0., 0., 0., 0.]

    risks = np.vstack((relative_risk(sel_MLE, beta, Sigma),
                       relative_risk(ind_est, beta, Sigma),
                       relative_risk(randomized_lasso_est, beta, Sigma),
                       relative_risk(randomized_rel_lasso_est, beta, Sigma),
                       relative_risk(rel_LASSO, beta, Sigma),
                       relative_risk(glm_LASSO, beta, Sigma)))

    partial_risks = np.vstack((partial_MLE_risk,
                               partial_ind_risk,
                               partial_randLasso_risk,
                               partial_relrandLasso_risk,
                               partial_relLasso_risk,
                               partial_Lasso_risk))

    naive_inf = np.vstack((cov_naive, length_naive, 0., nactive_LASSO, bias_naive, selective_naive_power, power_naive, power_naive_BH, fdr_naive_BH,
                           naive_discoveries.sum()))
    Lee_inf = np.vstack((cov_Lee, length_Lee, inf_entries, nactive_LASSO, bias_Lee, selective_Lee_power, power_Lee, power_Lee_BH, fdr_Lee_BH,
                         Lee_discoveries.sum()))
    Liu_inf = np.zeros((10, 1))
    MLE_inf = np.vstack((cov_MLE, length_MLE, 0., nonzero.sum(), bias_MLE, selective_MLE_power, power_MLE, power_MLE_BH, fdr_MLE_BH,
                         MLE_discoveries.sum()))
    nreport = np.vstack((Lee_nreport, 0., MLE_nreport))

    return np.vstack((risks, naive_inf, Lee_inf, Liu_inf, MLE_inf, partial_risks, nreport))


def comparison_cvmetrics_full(n=500, p=100, nval=500, rho=0.35, s=5, beta_type=1, snr=0.20,
                              randomizer_scale=np.sqrt(0.25), full_dispersion=True,
                              tuning_nonrand="lambda.min", tuning_rand="lambda.1se"):

    X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho, s=s, beta_type=beta_type, snr=snr)
    print("snr", snr)
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
    y = y - y.mean()
    true_set = np.asarray([u for u in range(p) if beta[u] != 0])

    if full_dispersion:
        dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
        sigma_ = np.sqrt(dispersion)
    else:
        dispersion = None
        sigma_ = np.std(y)
    print("estimated and true sigma", sigma, sigma_)

    lam_theory = sigma_ * 1. * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
    glm_LASSO_theory, glm_LASSO_1se, glm_LASSO_min, lam_min, lam_1se = glmnet_lasso(X, y, lam_theory/float(n))
    if tuning_nonrand == "lambda.min":
        lam_LASSO = lam_min
        glm_LASSO = glm_LASSO_min
    elif tuning_nonrand == "lambda.1se":
        lam_LASSO = lam_1se
        glm_LASSO = glm_LASSO_1se
    else:
        lam_LASSO = lam_theory/float(n)
        glm_LASSO = glm_LASSO_theory

    active_LASSO = (glm_LASSO != 0)
    nactive_LASSO = active_LASSO.sum()
    active_set_LASSO = np.asarray([r for r in range(p) if active_LASSO[r]])
    active_LASSO_bool = np.asarray([(np.in1d(active_set_LASSO[z], true_set).sum() > 0) for z in range(nactive_LASSO)],
                                   np.bool)

    rel_LASSO = np.zeros(p)
    Lee_nreport = 0
    bias_Lee = 0.
    bias_naive = 0.

    if nactive_LASSO > 0:
        rel_LASSO[active_LASSO] = np.linalg.pinv(X[:, active_LASSO]).dot(y)
        Lee_target = beta[active_LASSO]
        Lee_intervals, Lee_pval = selInf_R(X, y, glm_LASSO, n * lam_LASSO, sigma_, Type=1, alpha=0.1)

        if (Lee_pval.shape[0] == Lee_target.shape[0]):

            cov_Lee, selective_Lee_power = coverage(Lee_intervals, Lee_pval, Lee_target, beta[active_LASSO])
            inf_entries_bool = np.isinf(Lee_intervals[:, 1] - Lee_intervals[:, 0])
            inf_entries = np.mean(inf_entries_bool)
            if inf_entries == 1.:
                length_Lee = 0.
            else:
                length_Lee = np.mean((Lee_intervals[:, 1] - Lee_intervals[:, 0])[~inf_entries_bool])
            power_Lee = ((active_LASSO_bool) * (
                np.logical_or((0. < Lee_intervals[:, 0]), (0. > Lee_intervals[:, 1])))).sum() / float((beta != 0).sum())
            Lee_discoveries = BHfilter(Lee_pval, q=0.1)
            power_Lee_BH = (Lee_discoveries * active_LASSO_bool).sum() / float((beta != 0).sum())
            fdr_Lee_BH = (Lee_discoveries * ~active_LASSO_bool).sum() / float(max(Lee_discoveries.sum(), 1.))
            bias_Lee = np.mean(glm_LASSO[active_LASSO] - Lee_target)

            post_LASSO_OLS = np.linalg.pinv(X[:, active_LASSO]).dot(y)
            naive_sd = sigma_ * np.sqrt(np.diag((np.linalg.inv(X[:, active_LASSO].T.dot(X[:, active_LASSO])))))
            naive_intervals = np.vstack([post_LASSO_OLS - 1.65 * naive_sd,
                                         post_LASSO_OLS + 1.65 * naive_sd]).T
            naive_pval = 2 * ndist.cdf(np.abs(post_LASSO_OLS) / naive_sd)
            cov_naive, selective_naive_power = coverage(naive_intervals, naive_pval, Lee_target, beta[active_LASSO])
            length_naive = np.mean(naive_intervals[:, 1] - naive_intervals[:, 0])
            power_naive = ((active_LASSO_bool) * (
                np.logical_or((0. < naive_intervals[:, 0]), (0. > naive_intervals[:, 1])))).sum() / float(
                (beta != 0).sum())
            naive_discoveries = BHfilter(naive_pval, q=0.1)
            power_naive_BH = (naive_discoveries * active_LASSO_bool).sum() / float((beta != 0).sum())
            fdr_naive_BH = (naive_discoveries * ~active_LASSO_bool).sum() / float(max(naive_discoveries.sum(), 1.))
            bias_naive = np.mean(rel_LASSO[active_LASSO] - Lee_target)

            partial_Lasso_risk = (glm_LASSO[active_LASSO] - Lee_target).T.dot(glm_LASSO[active_LASSO] - Lee_target)
            partial_relLasso_risk = (post_LASSO_OLS - Lee_target).T.dot(post_LASSO_OLS - Lee_target)
        else:
            Lee_nreport = 1
            cov_Lee, length_Lee, inf_entries, power_Lee, power_Lee_BH, fdr_Lee_BH, selective_Lee_power = [0., 0., 0., 0., 0., 0., 0.]
            cov_naive, length_naive, power_naive, power_naive_BH, fdr_naive_BH, selective_naive_power  = [0., 0., 0., 0., 0., 0.]
            naive_discoveries = np.zeros(1)
            Lee_discoveries = np.zeros(1)
            partial_Lasso_risk, partial_relLasso_risk = [0., 0.]

    elif nactive_LASSO == 0:
        Lee_nreport = 1
        cov_Lee, length_Lee, inf_entries, power_Lee, power_Lee_BH, fdr_Lee_BH, selective_Lee_power = [0., 0., 0., 0., 0., 0., 0.]
        cov_naive, length_naive, power_naive, power_naive_BH, fdr_naive_BH, selective_naive_power = [0., 0., 0., 0., 0., 0.]
        naive_discoveries = np.zeros(1)
        Lee_discoveries = np.zeros(1)
        partial_Lasso_risk, partial_relLasso_risk = [0., 0.]

    lasso_Liu = ROSI.gaussian(X, y, n * lam_LASSO)
    print(type(lasso_Liu))
    Lasso_soln_Liu = lasso_Liu.fit()
    active_set_Liu = np.nonzero(Lasso_soln_Liu != 0)[0]
    nactive_Liu = active_set_Liu.shape[0]
    active_Liu_bool = np.asarray([(np.in1d(active_set_Liu[a], true_set).sum() > 0) for a in range(nactive_Liu)], np.bool)
    Liu_nreport = 0

    if nactive_Liu > 0:
        Liu_target = beta[Lasso_soln_Liu != 0]
        df = lasso_Liu.summary(level=0.90, compute_intervals=True, dispersion=dispersion)
        Liu_lower, Liu_upper, Liu_pval = np.asarray(df['lower_confidence']), \
                                         np.asarray(df['upper_confidence']), \
                                         np.asarray(df['pvalue'])
        Liu_intervals = np.vstack((Liu_lower, Liu_upper)).T
        cov_Liu, selective_Liu_power = coverage(Liu_intervals, Liu_pval, Liu_target, beta[Lasso_soln_Liu != 0])
        length_Liu = np.mean(Liu_intervals[:, 1] - Liu_intervals[:, 0])
        power_Liu = ((active_Liu_bool) * (np.logical_or((0. < Liu_intervals[:, 0]),
                                                        (0. > Liu_intervals[:, 1])))).sum() / float((beta != 0).sum())
        Liu_discoveries = BHfilter(Liu_pval, q=0.1)
        power_Liu_BH = (Liu_discoveries * active_Liu_bool).sum() / float((beta != 0).sum())
        fdr_Liu_BH = (Liu_discoveries * ~active_Liu_bool).sum() / float(max(Liu_discoveries.sum(), 1.))

    else:
        Liu_nreport = 1
        cov_Liu, length_Liu, power_Liu, power_Liu_BH, fdr_Liu_BH, selective_Liu_power = [0., 0., 0., 0., 0., 0.]
        Liu_discoveries = np.zeros(1)

    if tuning_rand == "lambda.min":
        randomized_lasso = lasso.gaussian(X,
                                          y,
                                          feature_weights= n * lam_min * np.ones(p),
                                          randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)
    elif tuning_rand == "lambda.1se":
        randomized_lasso = lasso.gaussian(X,
                                          y,
                                          feature_weights= n * lam_1se * np.ones(p),
                                          randomizer_scale= np.sqrt(n) * randomizer_scale * sigma_)
    else:
        randomized_lasso = lasso.gaussian(X,
                                          y,
                                          feature_weights= lam_theory * np.ones(p),
                                          randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)
    signs = randomized_lasso.fit()
    nonzero = signs != 0
    active_set_rand = np.asarray([t for t in range(p) if nonzero[t]])
    active_rand_bool = np.asarray([(np.in1d(active_set_rand[x], true_set).sum() > 0) for x in range(nonzero.sum())], np.bool)
    sel_MLE = np.zeros(p)
    ind_est = np.zeros(p)
    randomized_lasso_est = np.zeros(p)
    randomized_rel_lasso_est = np.zeros(p)
    MLE_nreport = 0

    if nonzero.sum() > 0:
        target_randomized = beta[nonzero]
        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = full_targets(randomized_lasso.loglike,
                                      randomized_lasso._W,
                                      nonzero,
                                      dispersion=dispersion)

        result = randomized_lasso.selective_MLE(observed_target,
                                                cov_target,
                                                cov_target_score)[0]

        MLE_estimate = result['MLE']
        ind_unbiased_estimator = result['unbiased']

        sel_MLE[nonzero] = MLE_estimate
        ind_est[nonzero] = ind_unbiased_estimator
        MLE_intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])
        MLE_pval = np.asarray(result['pvalue'])

        randomized_lasso_est = randomized_lasso.initial_soln
        randomized_rel_lasso_est = randomized_lasso._beta_full

        cov_MLE, selective_MLE_power = coverage(MLE_intervals, MLE_pval, target_randomized, beta[nonzero])
        length_MLE = np.mean(MLE_intervals[:, 1] - MLE_intervals[:, 0])
        power_MLE = ((active_rand_bool) * (np.logical_or((0. < MLE_intervals[:, 0]), (0. > MLE_intervals[:, 1])))).sum() / float((beta != 0).sum())
        MLE_discoveries = BHfilter(MLE_pval, q=0.1)
        power_MLE_BH = (MLE_discoveries * active_rand_bool).sum() / float((beta != 0).sum())
        fdr_MLE_BH = (MLE_discoveries * ~active_rand_bool).sum() / float(max(MLE_discoveries.sum(), 1.))
        bias_MLE = np.mean(MLE_estimate - target_randomized)

        partial_MLE_risk = (MLE_estimate - target_randomized).T.dot(MLE_estimate - target_randomized)
        partial_ind_risk = (ind_unbiased_estimator - target_randomized).T.dot(ind_unbiased_estimator - target_randomized)
        partial_randLasso_risk = (randomized_lasso_est[nonzero] - target_randomized).T.dot(randomized_lasso_est[nonzero] - target_randomized)
        partial_relrandLasso_risk = (randomized_rel_lasso_est[nonzero] - target_randomized).T.dot(randomized_rel_lasso_est[nonzero] - target_randomized)
    else:
        MLE_nreport = 1
        cov_MLE, length_MLE, power_MLE, power_MLE_BH, fdr_MLE_BH, bias_MLE, selective_MLE_power = [0., 0., 0., 0., 0., 0., 0.]
        MLE_discoveries = np.zeros(1)
        partial_MLE_risk, partial_ind_risk, partial_randLasso_risk, partial_relrandLasso_risk = [0., 0., 0., 0.]

    risks = np.vstack((relative_risk(sel_MLE, beta, Sigma),
                       relative_risk(ind_est, beta, Sigma),
                       relative_risk(randomized_lasso_est, beta, Sigma),
                       relative_risk(randomized_rel_lasso_est, beta, Sigma),
                       relative_risk(rel_LASSO, beta, Sigma),
                       relative_risk(glm_LASSO, beta, Sigma)))

    partial_risks = np.vstack((partial_MLE_risk,
                               partial_ind_risk,
                               partial_randLasso_risk,
                               partial_relrandLasso_risk,
                               partial_relLasso_risk,
                               partial_Lasso_risk))

    naive_inf = np.vstack((cov_naive, length_naive, 0., nactive_LASSO, bias_naive, selective_naive_power,
                           power_naive, power_naive_BH, fdr_naive_BH, naive_discoveries.sum()))
    Lee_inf = np.vstack((cov_Lee, length_Lee, inf_entries, nactive_LASSO, bias_Lee, selective_Lee_power,
                         power_Lee, power_Lee_BH, fdr_Lee_BH, Lee_discoveries.sum()))
    Liu_inf = np.vstack((cov_Liu, length_Liu, 0., nactive_Liu, bias_Lee, selective_Liu_power,
                         power_Liu, power_Liu_BH, fdr_Liu_BH, Liu_discoveries.sum()))
    MLE_inf = np.vstack((cov_MLE, length_MLE, 0., nonzero.sum(), bias_MLE, selective_MLE_power,
                         power_MLE, power_MLE_BH, fdr_MLE_BH, MLE_discoveries.sum()))
    nreport = np.vstack((Lee_nreport, Liu_nreport, MLE_nreport))

    return np.vstack((risks, naive_inf, Lee_inf, Liu_inf, MLE_inf, partial_risks, nreport))



def main(n=500, p=100, rho=0.35, s=5, beta_type=1, snr_values=np.array([0.15, 0.20, 0.31]),
         target="selected", tuning_nonrand="lambda.1se", tuning_rand="lambda.1se",
         randomizing_scale = np.sqrt(0.50), ndraw=20, outpath = None, plot=True):

    df_selective_inference = pd.DataFrame()
    df_risk = pd.DataFrame()

    if n > p:
        full_dispersion = True
    else:
        full_dispersion = False

    snr_list = []
    snr_list_0 = []
    for snr in snr_values:
        snr_list.append(snr*np.ones(4))
        snr_list_0.append(snr*np.ones(2))
        output_overall = np.zeros(55)
        if target == "selected":
            for i in range(ndraw):
                output_overall += np.squeeze(comparison_cvmetrics_selected(n=n,
                                                                           p=p,
                                                                           nval=n,
                                                                           rho=rho,
                                                                           s=s,
                                                                           beta_type=beta_type,
                                                                           snr=snr,
                                                                           randomizer_scale=randomizing_scale,
                                                                           full_dispersion=full_dispersion,
                                                                           tuning_nonrand =tuning_nonrand,
                                                                           tuning_rand=tuning_rand))
        elif target == "full":
            for i in range(ndraw):
                output_overall += np.squeeze(comparison_cvmetrics_full(n=n,
                                                                       p=p,
                                                                       nval=n,
                                                                       rho=rho,
                                                                       s=s,
                                                                       beta_type=beta_type,
                                                                       snr=snr,
                                                                       randomizer_scale=randomizing_scale,
                                                                       full_dispersion=full_dispersion,
                                                                       tuning_nonrand =tuning_nonrand,
                                                                       tuning_rand=tuning_rand))

        nLee = output_overall[52]
        nLiu = output_overall[53]
        nMLE = output_overall[54]

        relative_risk = (output_overall[0:6] / float(ndraw)).reshape((1, 6))
        partial_risk = np.hstack(((output_overall[46:50] / float(ndraw-nMLE)).reshape((1, 4)),
                                  (output_overall[50:52] / float(ndraw - nLee)).reshape((1, 2))))

        nonrandomized_naive_inf = np.hstack(((output_overall[6:12] / float(ndraw - nLee)).reshape((1, 6)),
                                             (output_overall[12:16] / float(ndraw)).reshape((1, 4))))
        nonrandomized_Lee_inf = np.hstack(((output_overall[16:22] / float(ndraw - nLee)).reshape((1, 6)),
                                          (output_overall[22:26] / float(ndraw)).reshape((1, 4))))
        nonrandomized_Liu_inf = np.hstack(((output_overall[26:32] / float(ndraw - nLiu)).reshape((1, 6)),
                                          (output_overall[32:36] / float(ndraw)).reshape((1, 4))))
        randomized_MLE_inf = np.hstack(((output_overall[36:42] / float(ndraw - nMLE)).reshape((1, 6)),
                                       (output_overall[42:46] / float(ndraw)).reshape((1, 4))))

        if target=="selected":
            nonrandomized_Liu_inf[nonrandomized_Liu_inf==0] = 'NaN'
        if target == "debiased":
            nonrandomized_Liu_inf[nonrandomized_Liu_inf == 0] = 'NaN'
            nonrandomized_Lee_inf[nonrandomized_Lee_inf == 0] = 'NaN'

        df_naive = pd.DataFrame(data=nonrandomized_naive_inf,columns=['coverage',
                                                                      'length',
                                                                      'prop-infty',
                                                                      'tot-active',
                                                                      'bias',
                                                                      'sel-power',
                                                                      'power',
                                                                      'power-BH',
                                                                      'fdr-BH',
                                                                      'tot-discoveries'])
        df_naive['method'] = "Naive"
        df_Lee = pd.DataFrame(data=nonrandomized_Lee_inf, columns=['coverage',
                                                                   'length',
                                                                   'prop-infty',
                                                                   'tot-active',
                                                                   'bias',
                                                                   'sel-power',
                                                                   'power',
                                                                   'power-BH',
                                                                   'fdr-BH',
                                                                   'tot-discoveries'])
        df_Lee['method'] = "Lee"

        df_Liu = pd.DataFrame(data=nonrandomized_Liu_inf,columns=['coverage',
                                                                  'length',
                                                                  'prop-infty',
                                                                  'tot-active',
                                                                  'bias',
                                                                  'sel-power',
                                                                  'power',
                                                                  'power-BH',
                                                                  'fdr-BH',
                                                                  'tot-discoveries'])
        df_Liu['method'] = "Liu"

        df_MLE = pd.DataFrame(data=randomized_MLE_inf, columns=['coverage',
                                                                'length',
                                                                'prop-infty',
                                                                'tot-active',
                                                                'bias',
                                                                'sel-power',
                                                                'power',
                                                                'power-BH',
                                                                'fdr-BH',
                                                                'tot-discoveries'])
        df_MLE['method'] = "MLE"

        df_risk_metrics = pd.DataFrame(data=relative_risk, columns=['sel-MLE',
                                                                    'ind-est',
                                                                    'rand-LASSO',
                                                                    'rel-rand-LASSO',
                                                                    'rel-LASSO',
                                                                    'LASSO'])
        df_risk_metrics['metric'] = "Full"
        df_prisk_metrics = pd.DataFrame(data=partial_risk,columns=['sel-MLE',
                                                                   'ind-est',
                                                                   'rand-LASSO',
                                                                   'rel-rand-LASSO',
                                                                   'rel-LASSO',
                                                                   'LASSO'])
        df_prisk_metrics['metric'] = "Partial"

        df_selective_inference = df_selective_inference.append(df_naive, ignore_index=True)
        df_selective_inference = df_selective_inference.append(df_Lee, ignore_index=True)
        df_selective_inference = df_selective_inference.append(df_Liu, ignore_index=True)
        df_selective_inference = df_selective_inference.append(df_MLE, ignore_index=True)

        df_risk = df_risk.append(df_risk_metrics, ignore_index=True)
        df_risk = df_risk.append(df_prisk_metrics, ignore_index=True)

    snr_list = list(itertools.chain.from_iterable(snr_list))
    df_selective_inference['n'] = n
    df_selective_inference['p'] = p
    df_selective_inference['s'] = s
    df_selective_inference['rho'] = rho
    df_selective_inference['beta-type'] = beta_type
    df_selective_inference['snr'] = pd.Series(np.asarray(snr_list))
    df_selective_inference['target'] = target

    snr_list_0 = list(itertools.chain.from_iterable(snr_list_0))
    df_risk['n'] = n
    df_risk['p'] = p
    df_risk['s'] = s
    df_risk['rho'] = rho
    df_risk['beta-type'] = beta_type
    df_risk['snr'] = pd.Series(np.asarray(snr_list_0))
    df_risk['target'] = target

    if outpath is None:
        outpath = os.path.dirname(__file__)

    outfile_inf_csv = os.path.join(outpath, "dims_" + str(n) + "_" + str(p) + "_inference_betatype" + str(beta_type) + target + "_rho_" + str(rho) + ".csv")
    outfile_risk_csv = os.path.join(outpath, "dims_" + str(n) + "_" + str(p) + "_risk_betatype" + str(beta_type) + target + "_rho_" + str(rho) + ".csv")
    outfile_inf_html = os.path.join(outpath, "dims_" + str(n) + "_" + str(p) + "_inference_betatype" + str(beta_type) + target + "_rho_" + str(rho) + ".html")
    outfile_risk_html = os.path.join(outpath, "dims_" + str(n) + "_" + str(p) + "_risk_betatype" + str(beta_type) + target + "_rho_" + str(rho) + ".html")
    df_selective_inference.to_csv(outfile_inf_csv, index=False)
    df_risk.to_csv(outfile_risk_csv, index=False)
    df_selective_inference.to_html(outfile_inf_html)
    df_risk.to_html(outfile_risk_html)

    if plot is True:
        plotRisk(df_risk)
        plotCoveragePower(df_selective_inference)


if __name__ == "__main__":
    main()

