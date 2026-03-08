"""PyMC model definition for the regime-switching HMM (v0)."""

import numpy as np
import pytensor
import pymc as pm
import pymc_extras as pmx
import pytensor.tensor as pt


def build_model(
    data: np.ndarray,
    K: int = 2,
    sticky_alpha_diag: float = 20.0,
    sticky_alpha_offdiag: float = 2.0,
    mu_prior_sigma: float = 0.05,
    lkj_eta: float = 2.0,
    sigma_prior_sigma: float = 0.10,
    order_means: bool = False,
) -> tuple[pm.Model, pm.Model]:
    """
    Build the unmarginalized and marginalized regime-switching HMM.

    Returns (unmarginalized_model, marginalized_model).
    The marginalized model integrates out the discrete chain for NUTS.

    When order_means=True, a soft ordering constraint is placed on the first
    asset's mean (mu[0,0] < mu[1,0] < ...) to break label symmetry.
    """
    T, d = data.shape

    sticky_alpha = np.full((K, K), sticky_alpha_offdiag)
    np.fill_diagonal(sticky_alpha, sticky_alpha_diag)

    with pm.Model() as model:
        P = pm.Dirichlet("P", a=sticky_alpha, shape=(K, K))
        init_dist = pm.Categorical.dist(p=np.ones(K) / K)
        chain = pmx.DiscreteMarkovChain("chain", P=P, init_dist=init_dist, shape=(T,))

        mu = pm.Normal("mu", mu=0.0, sigma=mu_prior_sigma, shape=(K, d))

        if order_means and K >= 2:
            for k in range(K - 1):
                pm.Potential(
                    f"mu_order_{k}",
                    pt.switch(mu[k + 1, 0] > mu[k, 0], 0.0, -1e10),
                )

        chols = []
        for k in range(K):
            chol_k, _, _ = pm.LKJCholeskyCov(
                f"chol_cov_{k}",
                n=d,
                eta=lkj_eta,
                sd_dist=pm.HalfNormal.dist(sigma=sigma_prior_sigma),
                compute_corr=True,
            )
            chols.append(chol_k)
        chol_stack = pt.stack(chols)  # (K, d, d)

        pm.MvNormal("obs", mu=mu[chain], chol=chol_stack[chain], observed=data)

    model_marg = pmx.marginalize(model, ["chain"])
    return model, model_marg


def build_model_manual(
    data: np.ndarray,
    K: int = 2,
    sticky_alpha_diag: float = 20.0,
    sticky_alpha_offdiag: float = 2.0,
    mu_prior_sigma: float = 0.05,
    lkj_eta: float = 2.0,
    sigma_prior_sigma: float = 0.10,
) -> pm.Model:
    """
    JAX-compatible HMM using a manual forward algorithm via pytensor.scan.

    Unlike build_model(), this avoids pmx.marginalize (which uses
    vectorize_graph and generates Alloc ops that break JAX tracing).
    Use this with nuts_sampler="numpyro".
    """
    T, d = data.shape

    sticky_alpha = np.full((K, K), sticky_alpha_offdiag)
    np.fill_diagonal(sticky_alpha, sticky_alpha_diag)

    with pm.Model() as model:
        P = pm.Dirichlet("P", a=sticky_alpha, shape=(K, K))
        log_P = pt.log(P)
        log_pi0 = pt.log(pt.ones(K) / K)

        mu = pm.Normal("mu", mu=0.0, sigma=mu_prior_sigma, shape=(K, d))

        chols = []
        for k in range(K):
            chol_k, _, _ = pm.LKJCholeskyCov(
                f"chol_cov_{k}",
                n=d,
                eta=lkj_eta,
                sd_dist=pm.HalfNormal.dist(sigma=sigma_prior_sigma),
                compute_corr=True,
            )
            chols.append(chol_k)
        chol_stack = pt.stack(chols)

        z_obs = pt.as_tensor_variable(data, name="z_obs")

        log_lik_components = []
        for k in range(K):
            dist_k = pm.MvNormal.dist(mu=mu[k], chol=chol_stack[k])
            log_lik_k = pm.logp(dist_k, z_obs)  # (T,)
            log_lik_components.append(log_lik_k)
        log_lik = pt.stack(log_lik_components, axis=1)  # (T, K)

        log_alpha_init = log_pi0 + log_lik[0]

        def forward_step(log_lik_t, log_alpha_prev, log_P_):
            return pt.logsumexp(log_alpha_prev[:, None] + log_P_, axis=0) + log_lik_t

        log_alphas = pytensor.scan(
            fn=forward_step,
            sequences=[log_lik[1:]],
            outputs_info=[log_alpha_init],
            non_sequences=[log_P],
            return_updates=False,
        )

        pm.Potential("hmm_loglik", pt.logsumexp(log_alphas[-1]))

    return model
