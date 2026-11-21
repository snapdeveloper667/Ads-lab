"""HOTRG for square periodic lattice gas on a 2**levels square."""
import numpy as np


def hotrg(levels, bond_dim, temperature, mu, coupling):
    """Return log Z per site, using two directional coarse-grainings per level."""
    n = np.arange(2)
    log_edge = coupling * np.outer(n, n) / temperature
    shift = log_edge.max()
    u, singular, vh = np.linalg.svd(np.exp(log_edge - shift))
    outgoing = u * np.sqrt(singular)
    incoming = vh.T * np.sqrt(singular)
    log_site = mu * n / temperature
    site_shift = log_site.max()
    tensor = np.einsum("s,sl,sr,su,sd->lrud", np.exp(log_site - site_shift),
                       incoming, outgoing, incoming, outgoing)
    scale = 2 * shift + site_shift
    for step in range(1, 2 * levels + 1):
        merged = np.einsum("abux,cdxv->acbduv", tensor, tensor, optimize=True)
        left, _, right, _, up, down = merged.shape
        merged = merged.reshape(left**2, right**2, up, down)
        left_matrix = merged.reshape(left**2, -1)
        right_matrix = merged.transpose(1, 0, 2, 3).reshape(right**2, -1)
        gram = left_matrix @ left_matrix.T + right_matrix @ right_matrix.T
        _, basis = np.linalg.eigh(gram)
        basis = basis[:, -min(bond_dim, len(basis)):]
        tensor = np.einsum("ai,bj,abuv->ijuv", basis, basis, merged, optimize=True)
        norm = np.max(np.abs(tensor))
        tensor = (tensor / norm).transpose(2, 3, 0, 1)
        scale += np.log(norm) / (2**step)
    trace = np.einsum("iijj->", tensor)
    if trace <= 0:
        raise ArithmeticError("Nonpositive tensor trace. Increase bond dimension.")
    log_z_density = scale + np.log(trace) / (4**levels)
    return {"log_partition_per_site": log_z_density,
            "grand_potential_per_site": -temperature * log_z_density}
