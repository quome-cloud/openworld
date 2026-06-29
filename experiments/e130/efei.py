"""Expert-Facilitated Extrospection-Introspection vs amateur search (Thm 4.4).

Expert model (Assumption 1): expert j returns X_j = theta* + b_j + eps_j, ||b_j||<=beta,
eps_j coordinates sub-Gaussian with variance proxy tau^2/n. Pooling N experts gives error
beta + tau*sqrt(d/(nN)) -> O(1/sqrt(N)), INDEPENDENT of behavior-pool size M.

Amateur model (Assumption 2): a behavior pool of size M with one resolving element; uniform
draws with replacement -> geometric hitting time with mean M = Theta(M)."""
import numpy as np


def expert_error(theta_star, N, n, tau, beta, d, rng):
    theta_star = np.asarray(theta_star, dtype=float)
    biases = rng.normal(size=(N, d)); biases *= (beta / max(np.linalg.norm(biases, axis=1).max(), 1e-9))
    noise = rng.normal(scale=tau / np.sqrt(n), size=(N, d))
    estimates = theta_star[None, :] + biases + noise
    return float(np.linalg.norm(estimates.mean(axis=0) - theta_star))


def amateur_trials(M, rng):
    # uniform draws with replacement until the single resolving behavior (index 0) is hit
    t = 0
    while True:
        t += 1
        if rng.integers(0, M) == 0:
            return t


def expert_consultations_for(delta, n, tau, beta, d):
    # tau*sqrt(d/(nN)) <= delta - beta  =>  N >= d*tau^2 / (n*(delta-beta)^2)   (Thm 4.4a)
    slack = max(delta - beta, 1e-9)
    return int(np.ceil(d * tau * tau / (n * slack * slack)))
