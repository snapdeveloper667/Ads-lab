"""Run one reproducible simulation and emit JSON."""
import argparse
import json
import math

from methods import metropolis, transfer
from models import GEOMETRIES, disks
from tensor import hotrg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=["transfer", "metropolis", "disks", "hotrg"])
    parser.add_argument("--geometry", choices=GEOMETRIES, default="square")
    for name, default in (("width", 4), ("height", 5), ("sweeps", 2000),
                          ("burn", 500), ("seed", 42), ("count", 36), ("levels", 3), ("bond-dim", 8)):
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in (("temperature", 1.), ("mu", -1.),
                          ("coupling", 1.), ("fraction", .4)):
        parser.add_argument("--" + name, type=float, default=default)
    p = parser.parse_args()
    if not all(map(math.isfinite, [p.temperature, p.mu, p.coupling, p.fraction])):
        parser.error("Parameters must be finite.")
    if min(p.width, p.height, p.count, p.sweeps) < 1 or min(p.burn, p.seed) < 0:
        parser.error("Invalid integer parameters.")
    if p.temperature <= 0 or not 0 < p.fraction < 1:
        parser.error("Require T > 0 and 0 < fraction < 1.")
    lattice_args = (p.width, p.height, p.geometry, p.temperature, p.mu, p.coupling)
    if p.method == "transfer":
        if p.width > 9:
            parser.error("Transfer matrix supports width <= 9 because memory grows as 4^width.")
        result = transfer(*lattice_args)
    elif p.method == "metropolis":
        result = metropolis(*lattice_args, p.sweeps, p.burn, p.seed)
    elif p.method == "hotrg":
        if p.geometry != "square" or not 1 <= p.levels <= 18 or not 1 <= p.bond_dim <= 12:
            parser.error("HOTRG requires square geometry, 1–18 levels and bond dimension 1–12.")
        result = hotrg(p.levels, p.bond_dim, p.temperature, p.mu, p.coupling)
    else:
        try:
            result = disks(p.count, p.fraction, p.sweeps, p.seed)
        except ValueError as exc:
            parser.error(str(exc))
    print(json.dumps({"parameters": vars(p), "result": result}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
