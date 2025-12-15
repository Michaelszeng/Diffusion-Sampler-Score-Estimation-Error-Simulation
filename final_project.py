"""
Toy verification of Theorem 2 (OU score-based sampling)
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from scipy.stats import gaussian_kde, multivariate_normal

np.random.seed(0)

# Mixture params
MEANS = [np.array([-1.5, -1.5]), np.array([1.5,  1.5])]
COVS  = [0.3 * np.eye(2), 0.3 * np.eye(2)]
WEIGHTS = np.array([0.5, 0.5])

# Default algorithm params
T_MAX_DEFAULT = 1.5
N_STEPS_DEFAULT = 50
NOISE_LEVEL_DEFAULT = 0.25

# Estimation params
N_P_SAMPLES = 1000     # samples from sampler p_T (for KDE)
N_TV_EVAL   = 4000     # eval points x~q for TV expectation
N_KL_MC     = 20000    # MC for KL(q||N(0,I))
N_RUNS      = 5         # repeats per setting for mean/stderr

# Utils
def sample_gaussian_mixture(means, covs, weights, n_samples):
    """Sample exactly n_samples from a Gaussian mixture."""
    K = len(means)
    counts = np.random.multinomial(n_samples, weights)
    chunks = []
    for k in range(K):
        if counts[k] > 0:
            chunks.append(np.random.multivariate_normal(means[k], covs[k], size=counts[k]))
    return np.vstack(chunks)

def logpdf_mixture(x, means, covs, weights):
    x = np.atleast_2d(x)
    comp = []
    for (m, c) in zip(means, covs):
        comp.append(multivariate_normal.logpdf(x, mean=m, cov=c))
    comp = np.stack(comp, axis=1)  # (n, K)
    return logsumexp(comp + np.log(weights)[None, :], axis=1)

def pdf_mixture(x, means, covs, weights):
    return np.exp(logpdf_mixture(x, means, covs, weights))

# Vectorized score of OU-evolved mixture q_t
def compute_score_at_time_t(x, t, means, covs, weights):
    """
    Vectorized: returns score nabla log q_t(x) for x shape (n,d).
    OU evolution:
      mu_j(t)=e^{-t} mu_j
      Sigma_j(t)=e^{-2t} Sigma_j + (1-e^{-2t}) I
    """
    x = np.atleast_2d(x)
    n, d = x.shape
    decay = np.exp(-t)
    decay2 = np.exp(-2*t)
    noise_var = 1.0 - decay2

    means_t = [decay * m for m in means]
    covs_t = [decay2 * C + noise_var * np.eye(d) for C in covs]

    # log component densities for all points
    logN = np.stack(
        [multivariate_normal.logpdf(x, mean=means_t[j], cov=covs_t[j]) for j in range(len(means_t))],
        axis=1
    )  # (n, K)

    # posterior weights: softmax(log w + logN)
    logw = np.log(weights)[None, :]
    log_num = logw + logN
    log_den = logsumexp(log_num, axis=1, keepdims=True)
    post = np.exp(log_num - log_den)  # (n, K)

    # component scores: -(Sigma^{-1})(x - mu)
    scores = np.zeros((n, d))
    for j in range(len(means_t)):
        inv_cov = np.linalg.inv(covs_t[j])
        diff = x - means_t[j][None, :]
        comp_score = -(diff @ inv_cov.T)  # (n,d)
        scores += post[:, j:j+1] * comp_score

    return scores

# Reverse sampler (discretized reverse OU)
def run_reverse_ou_sampler(n_samples, n_steps, T, means, covs, weights, perturb_std=0.0):
    """
    Discretized reverse OU sampler:
      X_{k+1} = X_k + (X_k + 2 s_{T-kh}(X_k)) h + sqrt(2h) Z_k
    with X_0 ~ N(0, I).
    """
    d = means[0].shape[0]
    h = T / n_steps

    x = np.random.randn(n_samples, d)  # gamma^d

    for k in range(n_steps):
        t_fwd = T - k * h
        score = compute_score_at_time_t(x, t_fwd, means, covs, weights)
        if perturb_std > 0:
            score = score + np.random.randn(*score.shape) * perturb_std

        drift = x + 2.0 * score
        x = x + drift * h + np.sqrt(2.0 * h) * np.random.randn(*x.shape)

    return x

# Theorem 2 terms
def estimate_kl_q_vs_standard_gaussian(means, covs, weights, n_mc=N_KL_MC):
    x = sample_gaussian_mixture(means, covs, weights, n_mc)
    log_q = logpdf_mixture(x, means, covs, weights)
    log_g = multivariate_normal.logpdf(x, mean=np.zeros(x.shape[1]), cov=np.eye(x.shape[1]))
    return float(np.mean(log_q - log_g))

def compute_m2(means, covs, weights):
    ex_norm2 = 0.0
    for w, m, C in zip(weights, means, covs):
        ex_norm2 += w * (np.trace(C) + float(np.dot(m, m)))
    return float(np.sqrt(ex_norm2))

def lipschitz_bound_score(covs):
    # Proxy bound: max_j ||Sigma_j(0)^{-1}||_op = 1 / min_j lambda_min(Sigma_j(0))
    lam_mins = [np.linalg.eigvalsh(C).min() for C in covs]
    return 1.0 / min(lam_mins)

def eps_score_from_iid_gaussian(perturb_std, d):
    # If score perturbation is iid N(0, perturb_std^2 I), then sqrt(E||noise||^2)=sqrt(d)*perturb_std
    return float(perturb_std * np.sqrt(d))

def theorem2_rhs(T, h, d, KL_q, L, m2, eps_score):
    term_forward = np.sqrt(max(KL_q, 0.0) * np.exp(-T))
    term_disc    = (L * np.sqrt(d * h) + L * m2 * h) * np.sqrt(T)
    term_score   = eps_score * np.sqrt(T)
    return term_forward + term_disc + term_score

# TV estimation
def estimate_tv_via_q_expectation(samples_p, means, covs, weights, n_eval=N_TV_EVAL, bw_method="scott", eps_floor=1e-300):
    """
    TV(p,q) = 0.5 ∫ |p-q| dx = 0.5 E_{x~q}[ |p(x)/q(x) - 1| ].
    Estimate p(x) by KDE from samples_p, and evaluate on fresh x~q.
    """
    kde = gaussian_kde(samples_p.T, bw_method=bw_method)
    x_eval = sample_gaussian_mixture(means, covs, weights, n_eval)
    p_hat = kde(x_eval.T)
    q_val = pdf_mixture(x_eval, means, covs, weights)
    q_val = np.maximum(q_val, eps_floor)
    return float(0.5 * np.mean(np.abs(p_hat / q_val - 1.0)))

# Experiment wrapper
def run_setting_tv(T, N, eps, n_p=N_P_SAMPLES, n_runs=N_RUNS):
    """
    Returns (mean_TV, stderr_TV) for given (T,N,eps).
    """
    tvs = []
    for _ in range(n_runs):
        p_samp = run_reverse_ou_sampler(n_p, N, T, MEANS, COVS, WEIGHTS, perturb_std=eps)
        tv = estimate_tv_via_q_expectation(p_samp, MEANS, COVS, WEIGHTS)
        tvs.append(tv)
    tvs = np.array(tvs)
    return float(tvs.mean()), float(tvs.std(ddof=1) / np.sqrt(len(tvs)))


def main():
    d = 2

    # Fixed theorem parameters (q-dependent)
    print("\n[Computing q-dependent parameters]")
    KL_q = estimate_kl_q_vs_standard_gaussian(MEANS, COVS, WEIGHTS)
    m2 = compute_m2(MEANS, COVS, WEIGHTS)
    L = lipschitz_bound_score(COVS)
    print(f"  KL(q||N)= {KL_q:.4f},  m2= {m2:.4f},  L(proxy)= {L:.4f}")

    # =========================================================
    # Discretization sweep: vary N, eps=0, T fixed
    # =========================================================
    T = T_MAX_DEFAULT
    Ns = [10, 20, 30, 50, 75, 100, 150]
    tv_mean_A, tv_se_A, rhs_A = [], [], []

    print("\nDiscretization sweep (eps=0, vary N)")
    for N in Ns:
        h = T / N
        tvm, tvse = run_setting_tv(T, N, eps=0.0)
        rhs = theorem2_rhs(T, h, d, KL_q, L, m2, eps_score=0.0)
        tv_mean_A.append(tvm); tv_se_A.append(tvse); rhs_A.append(rhs)
        print(f"  N={N:4d}, h={h:.4f}: TV={tvm:.4f} +/- {tvse:.4f}, RHS={rhs:.4f}, ratio={tvm/rhs:.3f}")

    # =========================================================
    # Score error sweep: vary eps, report TV vs eps*sqrt(T)
    # =========================================================
    N = N_STEPS_DEFAULT
    eps_list = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]
    tv_mean_B, tv_se_B, rhs_B = [], [], []
    print("\nScore error sweep (fix T,N; vary eps; report ΔTV)")

    tv0_mean, tv0_se = run_setting_tv(T, N, eps=0.0)
    print(f"  Baseline eps=0: TV0={tv0_mean:.4f} +/- {tv0_se:.4f}")

    for eps in eps_list:
        h = T / N
        tvm, tvse = (tv0_mean, tv0_se) if eps == 0.0 else run_setting_tv(T, N, eps=eps)
        eps_score = eps_score_from_iid_gaussian(eps, d)
        rhs = theorem2_rhs(T, h, d, KL_q, L, m2, eps_score=eps_score)
        tv_mean_B.append(tvm); tv_se_B.append(tvse); rhs_B.append(rhs)
        dTV = tvm - tv0_mean
        print(f"  eps={eps:>4.2f}: TV={tvm:.4f} +/- {tvse:.4f}, ΔTV={dTV:+.4f}, RHS={rhs:.4f}")

    # =========================================================
    # Time sweep: vary T with approximately fixed h
    # =========================================================
    print("\nTime sweep (eps=0, keep h approx fixed)")
    h_target = T_MAX_DEFAULT / N_STEPS_DEFAULT
    Ts = [0.5, 1.0, 1.5, 2.0, 3.0]
    tv_mean_C, tv_se_C, rhs_C, Ns_C = [], [], [], []

    for Tcur in Ts:
        Ncur = int(np.ceil(Tcur / h_target))
        hcur = Tcur / Ncur
        tvm, tvse = run_setting_tv(Tcur, Ncur, eps=0.0)
        rhs = theorem2_rhs(Tcur, hcur, d, KL_q, L, m2, eps_score=0.0)
        tv_mean_C.append(tvm); tv_se_C.append(tvse); rhs_C.append(rhs); Ns_C.append(Ncur)
        print(f"  T={Tcur:.2f}, N={Ncur:4d}, h={hcur:.4f}: TV={tvm:.4f} +/- {tvse:.4f}, RHS={rhs:.4f}")

    # =========================================================
    # Plots
    # =========================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # TV vs RHS as N changes
    axes[0].errorbar(Ns, tv_mean_A, yerr=tv_se_A, fmt="o-", label="TV(p,q)")
    axes[0].plot(Ns, rhs_A, "s--", label="RHS bound")
    axes[0].set_title("Discretization sweep (vary N)")
    axes[0].set_xlabel("N steps")
    axes[0].set_ylabel("Distance")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # ΔTV vs eps, plus eps*sqrt(T) trend line (scaled to match first nonzero)
    dTVs = [m - tv0_mean for m in tv_mean_B]
    axes[1].errorbar(eps_list, dTVs, yerr=tv_se_B, fmt="o-", label="ΔTV = TV(eps)-TV(0)")
    trend = np.array(eps_list) * np.sqrt(T)
    if len(eps_list) > 1 and eps_list[1] > 0:
        # scale trend to match magnitude at eps_list[1]
        scale = dTVs[1] / (trend[1] + 1e-12)
        axes[1].plot(eps_list, scale * trend, "k--", label="(scaled) eps*sqrt(T)")
    axes[1].set_title("Score error sweep (ΔTV scaling)")
    axes[1].set_xlabel("Score noise std (eps)")
    axes[1].set_ylabel("ΔTV")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    # TV vs T
    axes[2].errorbar(Ts, tv_mean_C, yerr=tv_se_C, fmt="o-", label="TV(p,q)")
    axes[2].plot(Ts, rhs_C, "s--", label="RHS bound")
    axes[2].set_title("Time sweep (vary T, h≈const)")
    axes[2].set_xlabel("T")
    axes[2].set_ylabel("Distance")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("theorem2_scaling_sweeps.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
