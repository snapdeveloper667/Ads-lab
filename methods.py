"""Grand canonical single-component lattice gas, H = -J B, k_B = 1."""
import numpy as np
from scipy.special import logsumexp

from models import lattice


def transfer(width, height, geometry, temperature, mu, coupling):
    """Exact row transfer for the finite open strip. Propagate moments in log space."""
    states = ((np.arange(2**width)[:, None] >> np.arange(width)) & 1).astype(float)
    number = states.sum(axis=1)
    bonds = (states[:, :-1] * states[:, 1:]).sum(axis=1)
    local = (mu * number + coupling * bonds) / temperature
    log_weight = local.copy()
    occupancy_mean = number.copy()
    occupancy_variance = np.zeros_like(number)
    bond_mean = bonds.copy()
    for row in range(height - 1):
        cross = np.zeros((len(states), len(states)))
        for col in range(width):
            occupancy = states[:, col]
            if geometry != "honeycomb" or (row + col) % 2 == 0:
                cross += np.outer(occupancy, occupancy)
            if geometry == "triangular" and col + 1 < width:
                cross += np.outer(occupancy, states[:, col + 1])
        log_joint = log_weight[:, None] + coupling * cross / temperature + local
        log_weight = logsumexp(log_joint, axis=0)
        conditional = np.exp(log_joint - log_weight)
        next_mean = conditional.T @ occupancy_mean
        occupancy_variance = (conditional * (occupancy_variance[:, None]
                             + (occupancy_mean[:, None] - next_mean)**2)).sum(axis=0)
        occupancy_mean = next_mean + number
        bond_mean = (conditional * (bond_mean[:, None] + cross)).sum(axis=0) + bonds
    log_z = logsumexp(log_weight)
    weights = np.exp(log_weight - log_z)
    mean_n = weights @ occupancy_mean
    variance_n = weights @ (occupancy_variance + (occupancy_mean - mean_n)**2)
    sites = width * height
    return {"coverage": mean_n / sites, "energy_per_site": -coupling * (weights @ bond_mean) / sites,
            "coverage_response": variance_n / (temperature * sites), "log_partition": log_z,
            "grand_potential_per_site": -temperature * log_z / sites}


def metropolis(width, height, geometry, temperature, mu, coupling, sweeps, burn, seed):
    """Single-site updates. Measurements are correlated across sweeps."""
    rng = np.random.default_rng(seed)
    sites = width * height
    edges = lattice(width, height, geometry)
    neighbors = [[] for _ in range(sites)]
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)
    occupied = np.zeros(sites, dtype=int)
    count = pairs = accepted = 0
    measurements = []
    for sweep in range(sweeps + burn):
        for i in rng.integers(sites, size=sites):
            change = 1 - 2 * occupied[i]
            pair_change = change * occupied[neighbors[i]].sum(dtype=int)
            delta = -coupling * pair_change - mu * change
            if delta <= 0 or rng.random() < np.exp(-delta / temperature):
                occupied[i] += change
                count += change
                pairs += pair_change
                accepted += sweep >= burn
        if sweep >= burn:
            measurements.append((count, pairs))
    counts, bonds = np.asarray(measurements).T
    return {"coverage": counts.mean() / sites, "energy_per_site": -coupling * bonds.mean() / sites,
            "coverage_response": counts.var() / (temperature * sites),
            "acceptance": accepted / (sweeps * sites),
            "configuration": occupied.reshape(height, width).tolist()}
