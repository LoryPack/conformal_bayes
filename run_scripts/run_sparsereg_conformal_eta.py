import time

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.scipy.stats import norm
from sklearn.linear_model import LassoCV, Lasso
from tqdm import tqdm

from conformal_bayes import conformal_Bayes_functions as cb
# import from cb package
from run_scripts.load_data import load_traintest_sparsereg


# Main run function for sparse regression
def run_sparsereg_conformal(dataset, misspec=False, eta: float = 1):
    # Compute intervals
    # Initialize
    train_frac = 0.7
    x, y, x_test, y_test, y_plot, n, d = load_traintest_sparsereg(train_frac, dataset, 100)

    # Load posterior samples
    suffix = dataset
    if misspec:
        suffix = dataset + "_misspec"

    beta_post = jnp.load("samples/beta_post_sparsereg_{}.npy".format(suffix))
    intercept_post = jnp.load("samples/intercept_post_sparsereg_{}.npy".format(suffix))
    sigma_post = jnp.load("samples/sigma_post_sparsereg_{}.npy".format(suffix))

    # Initialize
    alpha = 0.2
    rep = np.shape(beta_post)[0]
    n_test = np.shape(x_test)[0]

    coverage_cb = np.zeros((rep, n_test))
    coverage_cb_exact = np.zeros((rep, n_test))  # avoiding grid effects

    length_cb = np.zeros((rep, n_test))

    region_cb = np.zeros((rep, n_test, np.shape(y_plot)[0]))

    times_cb = np.zeros(rep)

    for j in tqdm(range(rep)):
        seed = 100 + j
        # load dataset
        x, y, x_test, y_test, y_plot, n, d = load_traintest_sparsereg(train_frac, dataset, seed)
        dy = y_plot[1] - y_plot[0]

        # Conformal Bayes
        start = time.time()

        @jit  # normal loglik from posterior samples
        def normal_loglikelihood(y, x):
            return norm.logpdf(y, loc=jnp.dot(beta_post[j], x.transpose()) + intercept_post[j],
                               scale=sigma_post[j])  # compute likelihood samples

        print(eta)
        logp_samp_n = normal_loglikelihood(y, x)
        logwjk = normal_loglikelihood(y_plot.reshape(-1, 1, 1), x_test)
        logwjk_test = normal_loglikelihood(y_test, x_test).reshape(1, -1, n_test)

        for i in (range(n_test)):
            region_cb[j, i] = cb.compute_cb_region_IS(alpha, logp_samp_n, logwjk[:, :, i], eta)
            coverage_cb[j, i] = region_cb[j, i, np.argmin(np.abs(y_test[i] - y_plot))]  # grid coverage
            length_cb[j, i] = np.sum(region_cb[j, i]) * dy
        end = time.time()
        times_cb[j] = end - start

        # compute exact coverage to avoid grid effects
        for i in (range(n_test)):
            coverage_cb_exact[j, i] = cb.compute_cb_region_IS(alpha, logp_samp_n,
                                                              logwjk_test[:, :, i], eta)  # exact coverage

    # #Save regions (need to update)
    if eta != 1:
        suffix = suffix + "_eta_" + str(eta)

    np.save("results/region_cb_sparsereg_{}".format(suffix), region_cb)

    np.save("results/coverage_cb_sparsereg_{}".format(suffix), coverage_cb)
    np.save("results/coverage_cb_exact_sparsereg_{}".format(suffix), coverage_cb_exact)

    np.save("results/length_cb_sparsereg_{}".format(suffix), length_cb)
    np.save("results/times_cb_sparsereg_{}".format(suffix), times_cb)
    return np.mean(length_cb), np.mean(coverage_cb), np.mean(coverage_cb_exact)


if __name__ == "__main__":
    misspec_list = [True, False]
    # misspec_list = [True]
    eta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    for misspec in misspec_list:
        for eta in eta_list:
            mean_length, mean_coverage, mean_coverage_exact = run_sparsereg_conformal('diabetes', misspec, eta=eta)
            print("eta: {}; Mean length: {}; Mean coverage: {}; Mean coverage exact: {}".format(eta, mean_length,
                                                                                                mean_coverage,
                                                                                                mean_coverage_exact))
