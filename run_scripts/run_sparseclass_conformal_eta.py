import time
import numpy as np
from tqdm import tqdm
import pandas as pd
import jax.numpy as jnp
from jax import jit
from jax.scipy.stats import norm
import jax.scipy as jsp
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

# import from cb package
from run_scripts.load_data import load_traintest_sparseclass
from conformal_bayes import conformal_Bayes_functions as cb
from conformal_bayes import Bayes_MCMC_functions as bmcmc


# Main run function for sparse classification
def run_sparseclass_conformal(dataset, eta: float = 1):
    # Compute intervals
    # Load posterior samples
    beta_post = jnp.load("samples/beta_post_sparseclass_{}.npy".format(dataset))
    intercept_post = jnp.load("samples/intercept_post_sparseclass_{}.npy".format(dataset))

    # Initialize
    train_frac = 0.7
    x, y, x_test, y_test, y_plot, n, d = load_traintest_sparseclass(train_frac, dataset, 100)

    alpha = 0.2
    rep = np.shape(beta_post)[0]
    n_test = np.shape(x_test)[0]

    coverage_cb = np.zeros((rep, n_test))

    length_cb = np.zeros((rep, n_test))

    region_cb = np.zeros((rep, n_test, 2))

    times_cb = np.zeros(rep)

    for j in tqdm(range(rep)):
        seed = 100 + j

        # load data
        x, y, x_test, y_test, y_plot, n, d = load_traintest_sparseclass(train_frac, dataset, seed)

        @jit
        def logistic_loglikelihood(y, x):
            eta = (jnp.dot(beta_post[j], x.transpose()) + intercept_post[j])
            B = np.shape(eta)[0]
            n = np.shape(eta)[1]
            eta = eta.reshape(B, n, 1)
            temp0 = np.zeros((B, n, 1))
            logp = -jsp.special.logsumexp(jnp.concatenate((temp0, -eta), axis=2), axis=2)  # numerically stable
            log1p = -jsp.special.logsumexp(jnp.concatenate((temp0, eta), axis=2), axis=2)
            return y * logp + (1 - y) * log1p  # compute likelihood samples

        print(eta)
        # Conformal Bayes
        start = time.time()
        logp_samp_n = logistic_loglikelihood(y, x)
        logwjk = logistic_loglikelihood(y_plot.reshape(-1, 1, 1), x_test)
        # conformal
        for i in (range(n_test)):
            region_cb[j, i] = cb.compute_cb_region_IS(alpha, logp_samp_n, logwjk[:, :, i], eta)
            coverage_cb[j, i] = region_cb[j, i, np.argmin(np.abs(y_test[i] - y_plot))]
            length_cb[j, i] = np.sum(region_cb[j, i])
        end = time.time()
        times_cb[j] = end - start

    # Save regions (need to update)
    suffix = dataset
    if eta != 1:
        suffix = dataset + "_eta_" + str(eta)

    np.save("results/region_cb_sparseclass_{}".format(suffix), region_cb)
    np.save("results/coverage_cb_sparseclass_{}".format(suffix), coverage_cb)
    np.save("results/length_cb_sparseclass_{}".format(suffix), length_cb)
    np.save("results/times_cb_sparseclass_{}".format(suffix), times_cb)

    return np.mean(length_cb), np.mean(coverage_cb)


if __name__ == "__main__":
    # misspec_list = [True]
    eta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    for eta in eta_list:
        mean_length, mean_coverage = run_sparseclass_conformal('breast', eta=eta)
        print("eta: {}; Mean length: {}; Mean coverage: {}".format(eta, mean_length, mean_coverage, ))
