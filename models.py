"""Open lattice strips and periodic hard disks."""
import numpy as np

GEOMETRIES = ("square", "triangular", "honeycomb")


def lattice(width, height, geometry):
    """Brick-wall embedding for honeycomb; all boundaries are open."""
    horizontal = [(row * width + col, row * width + col + 1)
                  for row in range(height) for col in range(width - 1)]
    vertical = []
    for row in range(height - 1):
        for col in range(width):
            site = row * width + col
            if geometry != "honeycomb" or (row + col) % 2 == 0:
                vertical.append((site, site + width))
            if geometry == "triangular" and col + 1 < width:
                vertical.append((site, site + width + 1))
    return horizontal + vertical


def disks(count, fraction, sweeps, seed):
    """Metropolis sampling in an orthorhombic box, disk diameter = 1."""
    rng = np.random.default_rng(seed)
    columns = int(np.ceil(np.sqrt(count)))
    rows = (count + columns - 1) // columns
    spacing = np.sqrt(count * np.pi / (4 * fraction * columns * rows))
    if spacing < 1 or min(columns, rows) * spacing < 1:
        raise ValueError("Packing fraction is too high for square-grid initialization.")
    box = spacing * np.array([columns, rows])
    xy = spacing * np.array([(i % columns, i // columns) for i in range(count)],
                            dtype=float)
    accepted = 0
    for _ in range(sweeps * count):
        i = rng.integers(count)
        trial = (xy[i] + rng.uniform(-.3, .3, 2)) % box
        displacement = xy - trial
        displacement -= box * np.rint(displacement / box)
        distance_sq = np.sum(displacement ** 2, axis=1)
        distance_sq[i] = np.inf
        if np.all(distance_sq >= 1):
            xy[i] = trial
            accepted += 1
    return {"packing_fraction": fraction, "number_density": count / np.prod(box),
            "acceptance": accepted / (sweeps * count), "box": box.tolist(),
            "positions": xy.tolist()}
