import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Set the seeds for both PyTorch and NumPy to ensure reproducibility of the results. And start a timer to measure the execution time of the code.
rng = np.random.default_rng(0)
torch.manual_seed(0)
t_start = time.perf_counter()

# Set up the two squares distribution. The two squares are defined in a 2D space, and the DARK_CORNERS variable contains the coordinates of the corners of the dark squares. The sample_pinit function samples from a standard normal distribution, while the sample_pdata function samples from the two squares distribution. The sample_marginal_path function generates samples along a path between the two distributions based on a time parameter t.
_I, _J = np.indices((2, 2))
_DARK = (_I + _J) % 2 == 0
DARK_CORNERS = np.stack([-2 + 2 * _J[_DARK], -2 + 2 * _I[_DARK]], axis=1).astype(float)

# Noise distribution: p_init = N(0, I)
def sample_pinit(n, rng):
    return rng.standard_normal((n, 2))

# Data distribution: p_data = uniform over two squares
def sample_pdata(n, rng):
    idx = rng.integers(0, len(DARK_CORNERS), n)
    return DARK_CORNERS[idx] + rng.uniform(0, 2, (n, 2))

# Marginal path: p_t = t * p_data + (1-t) * p_init
def sample_marginal_path(t, n, rng):
    alpha = t
    beta = 1 - t
    z = sample_pdata(n, rng)
    eps = sample_pinit(n, rng)
    return alpha * z + beta * eps

# From here up until the algo 3 it's the creation of the neural network u_theta(x, t) 
# Network u_theta(x, t): input (x, t) in R^3, output a vector in R^2
model = nn.Sequential(
    nn.Linear(3, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, 2),
)

def u_theta(model, x, t):
    return model(torch.cat([x, t], dim=1))


# Training (Algorithm 3): conditional flow matching
B, steps = 2048, 4000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1 / steps))
losses = []
for step in range(1, steps + 1):
    z = torch.from_numpy(sample_pdata(B, rng)).float()    # z ~ p_data
    eps = torch.from_numpy(sample_pinit(B, rng)).float()  # eps ~ N(0, I)
    t = torch.rand(B, 1)                                   # t ~ Unif[0, 1]
    x = t * z + (1 - t) * eps                              # Algorithm 3: x = t*z + (1-t)*eps
    target = z - eps                                       # u_t(x|z) = z - eps for CondOT
    loss = ((u_theta(model, x, t) - target) ** 2).sum(dim=1).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    losses.append(loss.item())
    if step % 500 == 0:
        print(f"step {step:5d}  avg loss {np.mean(losses[-500:]):.4f}")
train_time = time.perf_counter() - t_start

# Sampling (Algorithm 1): Euler method X_{t+h} = X_t + h * u_theta(X_t, t)
n_samples, n_steps = 100_000, 100
h = 1 / n_steps
snap_steps = [0, 25, 50, 75, 100]
with torch.no_grad():
    X = torch.from_numpy(sample_pinit(n_samples, rng)).float()  # X_0 ~ p_init
    snapshots = [X.numpy().copy()]
    for i in range(n_steps):
        t = torch.full((n_samples, 1), i * h)
        X = X + h * u_theta(model, X, t)
        if i + 1 in snap_steps:
            snapshots.append(X.numpy().copy())

# Loss curve: raw loss plus a 100-step moving average
plt.figure(figsize=(8, 4))
plt.plot(losses, alpha=0.3, label="loss")
plt.plot(np.arange(99, steps), np.convolve(losses, np.ones(100) / 100, mode="valid"), label="moving avg (100)")
plt.xlabel("step")
plt.ylabel("loss")
plt.legend()
plt.savefig("loss_curve_two_squares.png", dpi=120)

# Comparison figure: ground truth path (top) vs learned Euler samples (bottom)
lim = 3
ts = [s / n_steps for s in snap_steps]
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for j, t in enumerate(ts):
    rows = [sample_marginal_path(t, 100_000, rng), snapshots[j]]
    for r, pts in enumerate(rows):
        ax = axes[r, j]
        ax.hist2d(pts[:, 0], pts[:, 1], bins=150, range=[[-lim, lim], [-lim, lim]], cmap="viridis")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t = {t:.2f}")
axes[0, 0].set_ylabel("Ground truth", fontsize=14)
axes[1, 0].set_ylabel("Learned", fontsize=14)
plt.tight_layout()
plt.savefig("flow_two_squares.png", dpi=120)

total_time = time.perf_counter() - t_start
print(f"training time: {train_time:.1f} s")
print(f"total time:    {total_time:.1f} s")
plt.show()