"""SHU formalism update operators over the stereotype space S = R^d (Defs 2.6-2.7, 4.5).
The two theorems (4.4 EFEI separation, 4.6 cycle convergence) are validated against these."""
import numpy as np


def tension(sigma_I, sigma_E):
    """T(H) = ||sigma_I - sigma_E||  (Def 2.6)."""
    return float(np.linalg.norm(np.asarray(sigma_I) - np.asarray(sigma_E)))


def E_alpha(sigma_E, X, alpha):
    """Extrospection synthesis: assimilate reading X into the extrospective stereotype (Def 2.7)."""
    return (1.0 - alpha) * np.asarray(sigma_E) + alpha * np.asarray(X)


def I_gamma(sigma_I, sigma_E, gamma):
    """Introspection retrosynthesis: consolidate the reading into the introspective stereotype (Def 2.7)."""
    return (1.0 - gamma) * np.asarray(sigma_I) + gamma * np.asarray(sigma_E)


def cycle_map(sigma_I, sigma_E, theta_star, alpha, gamma):
    """One BSTC behavioral cycle (Def 4.5): extrospection synthesis against the true context,
    then introspection retrosynthesis against the fresh reading."""
    sE_next = E_alpha(sigma_E, theta_star, alpha)
    sI_next = I_gamma(sigma_I, sE_next, gamma)
    return sI_next, sE_next


def rho(alpha, gamma):
    """Contraction modulus of the cycle map (Thm 4.6): spectral radius max(1-gamma, 1-alpha)."""
    return max(1.0 - gamma, 1.0 - alpha)
