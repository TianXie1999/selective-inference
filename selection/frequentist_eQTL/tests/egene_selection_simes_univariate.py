from __future__ import print_function
from scipy.stats import norm as normal
import numpy as np
import os
import sys
import regreg.api as rr
import statsmodels.api as sm

from scipy.stats import f
from scipy.stats.stats import pearsonr


def simes_selection_egene(X,
                          y,
                          randomizer='gaussian',
                          randomization_scale=1.):
    n, p = X.shape

    T_stats = np.zeros(p)
    for k in range(p):
        T_stats[k] = pearsonr(X[:, k], y)[0]

    if randomizer == 'gaussian':

        perturb = np.random.standard_normal(p)

        randomized_T_stats = np.multiply(np.sign(T_stats), np.sqrt(
            (n - 22.) * np.true_divide(T_stats ** 2., 1. - T_stats ** 2.))) + randomization_scale * perturb

        p_val_randomized = np.sort(2. * (1. - normal.cdf(np.true_divide(np.abs(randomized_T_stats),
                                                                        np.sqrt(1. + (randomization_scale ** 2))))))

        indices_order = np.argsort(2. * (1. - normal.cdf(np.true_divide(np.abs(randomized_T_stats),
                                                                        np.sqrt(1. + (randomization_scale ** 2))))))

    elif randomizer == 'none':

        randomized_T_stats = (n - 22.) * np.true_divide(T_stats ** 2., 1. - T_stats ** 2.)

        p_val_randomized = np.sort(1. - f.cdf(np.true_divide(np.abs(randomized_T_stats), np.sqrt(1.)), 1, n - 2))

        indices_order = np.argsort(1. - f.cdf(np.true_divide(np.abs(randomized_T_stats), np.sqrt(1.)), 1, n - 2))

    simes_p_randomized = np.min((p / (np.arange(p) + 1.)) * p_val_randomized)

    i_0 = np.argmin((p / (np.arange(p) + 1.)) * p_val_randomized)

    t_0 = indices_order[i_0]

    sigma_hat = np.sqrt((1. - (T_stats[t_0] ** 2)) * np.var(y)) / np.sqrt(n - 22.)

    T_stats_active = T_stats[t_0]

    u_1 = ((i_0 + 1.) / p) * np.min(
        np.delete((p / (np.arange(p) + 1.)) * p_val_randomized, i_0))

    if i_0 > p - 2:
        u_2 = -1
    else:
        u_2 = p_val_randomized[i_0 + 1]

    if i_0 == 0:
        u_3 = -1
    else:
        u_3 = p_val_randomized[i_0 - 1]

    return simes_p_randomized, i_0, t_0, u_1, u_2, u_3, np.sign(randomized_T_stats[t_0]), sigma_hat, np.sign(
        T_stats_active)


if __name__ == "__main__":

    path = sys.argv[1]
    outdir = sys.argv[2]
    result = sys.argv[3]

    outfile = os.path.join(outdir, "randomized_Z_simes_" + str(result) + ".txt")

    gene_file = path + "Genes.txt"

    with open(gene_file) as g:
        content = g.readlines()

    content = [x.strip() for x in content]
    sys.stderr.write("length" + str(len(content)) + "\n")

    iter = int(len(content))
    output = np.zeros((iter, 10))

    for j in range(iter):
        X = np.load(os.path.join(path + "X_" + str(content[j])) + ".npy")
        n, p = X.shape
        # X -= X.mean(0)[None, :]
        # X /= X.std(0)[None, :]
        # * np.sqrt(n))

        y = np.load(os.path.join(path + "y_" + str(content[j])) + ".npy")
        y = y.reshape((y.shape[0],))

        sys.stderr.write("iteration completed" + str(j) + "\n")
        # simes = simes_selection_egene(X, y, randomizer= 'none')
        simes = simes_selection_egene(X, y, randomizer='gaussian', randomization_scale=0.7)

        output[j, 0] = p
        output[j, 1:] = simes

        # beta = np.load(os.path.join(path + "b_" + str(content[j])) + ".npy")

    np.savetxt(outfile, output)