"""Periodic kagome nearest-neighbor exclusion benchmark."""
import argparse
import itertools
import json
import math
import random


def edges(width, height):
    """Build the line graph of a periodic honeycomb lattice."""
    incident = [[] for _ in range(2 * width * height)]
    count = width * height
    for y, x in itertools.product(range(height), range(width)):
        a = y * width + x
        targets = (a, y * width + (x + 1) % width,
                   ((y + 1) % height) * width + x)
        for direction, b in enumerate(targets):
            site = 3 * a + direction
            incident[a].append(site)
            incident[count + b].append(site)
    return sorted({tuple(sorted(pair)) for group in incident
                   for pair in itertools.combinations(group, 2)})


def exact(sites, neighbors, log_activity):
    counts = [0] * (sites + 1)

    def visit(remaining, occupied):
        if not remaining:
            counts[occupied] += 1
            return
        bit = remaining & -remaining
        site = bit.bit_length() - 1
        rest = remaining ^ bit
        visit(rest, occupied)
        visit(rest & ~neighbors[site], occupied + 1)

    visit((1 << sites) - 1, 0)
    log_terms = [(n, math.log(c) + n * log_activity)
                 for n, c in enumerate(counts) if c]
    shift = max(value for _, value in log_terms)
    weights = [(n, math.exp(value - shift)) for n, value in log_terms]
    total = sum(weight for _, weight in weights)
    return {"coverage": sum(n * w for n, w in weights) / (total * sites),
            "log_partition": shift + math.log(total),
            "coefficients": counts[:log_terms[-1][0] + 1]}


def monte_carlo(sites, neighbors, log_activity, sweeps, burn, seed):
    rng = random.Random(seed)
    occupied = 0
    total = 0
    insert = math.exp(min(0, log_activity))
    remove = math.exp(min(0, -log_activity))
    for sweep in range(burn + sweeps):
        for _ in range(sites):
            site = rng.randrange(sites)
            bit = 1 << site
            if occupied & bit:
                if rng.random() < remove:
                    occupied ^= bit
            elif not occupied & neighbors[site] and rng.random() < insert:
                occupied |= bit
        if sweep >= burn:
            total += occupied.bit_count()
    return {"coverage": total / (sites * sweeps), "seed": seed,
            "sweeps": sweeps, "burn": burn}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("method", choices=("exact", "metropolis"))
    for key, default in (("width", 2), ("height", 2), ("sweeps", 10000),
                         ("burn", 1000), ("seed", 42)):
        p.add_argument("--" + key, type=int, default=default)
    p.add_argument("--temperature", type=float, default=1.)
    p.add_argument("--mu", type=float, default=0.)
    a = p.parse_args()
    if min(a.width, a.height) < 2 or a.sweeps < 1 or a.burn < 0:
        p.error("Use width and height >= 2, sweeps >= 1 and burn >= 0.")
    if not math.isfinite(a.temperature) or not math.isfinite(a.mu) or a.temperature <= 0:
        p.error("Use finite parameters and temperature > 0.")
    sites = 3 * a.width * a.height
    if a.method == "exact" and sites > 24:
        p.error("Exact enumeration is limited to 24 sites.")
    log_activity = a.mu / a.temperature
    if not math.isfinite(log_activity) or abs(log_activity) > 1e300:
        p.error("Chemical potential divided by temperature is out of range.")
    bonds = edges(a.width, a.height)
    neighbors = [0] * sites
    for u, v in bonds:
        neighbors[u] |= 1 << v
        neighbors[v] |= 1 << u
    result = (exact(sites, neighbors, log_activity) if a.method == "exact"
              else monte_carlo(sites, neighbors, log_activity, a.sweeps, a.burn, a.seed))
    print(json.dumps({"sites": sites, "edges": len(bonds), **result}, indent=2))


if __name__ == "__main__":
    main()
