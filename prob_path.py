import numpy as np
import matplotlib.pyplot as plt

# Lower-left corners of the 32 dark squares of an 8x8 board on [-4, 4]^2.
# Square (i, j) has corner (-4 + j, -4 + i) and is dark when (i + j) is even.
_I, _J = np.indices((8, 8))
_DARK = (_I + _J) % 2 == 0
DARK_CORNERS = np.stack([-4 + _J[_DARK], -4 + _I[_DARK]], axis=1).astype(float)

def sample_pinit(n, rng):
    return rng.standard_normal((n, 2))

def sample_pdata(n, rng):
    idx = rng.integers(0, len(DARK_CORNERS), n)
    return DARK_CORNERS[idx] + rng.uniform(0, 1, (n, 2))

def sample_marginal_path(t, n, rng):
    alpha = t
    beta = 1 - t
    z = sample_pdata(n, rng)
    eps = sample_pinit(n, rng)
    return alpha * z + beta * eps


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    n = 100_000

    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    for ax, t in zip(axes, times):
        x = sample_marginal_path(t, n, rng)
        ax.hist2d(x[:, 0], x[:, 1], bins=200, range=[[-5, 5], [-5, 5]], cmap="viridis")
        ax.set_title(f"t = {t:.2f}")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Marginal probability path (Gaussian, alpha_t = t, beta_t = 1 - t)")
    plt.savefig("marginal_path.png", dpi=150)
    plt.show()
