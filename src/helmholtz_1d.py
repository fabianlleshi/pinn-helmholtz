import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import quad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

K = 0.0
N_MODES = 200
N_COLLOCATION = 200
MU = 100.0
N_ITER = 8000


def f_np(x):
    return np.exp(-x**2 / 20.0)


def f_torch(x):
    return torch.exp(-x**2 / 20.0)


n = np.arange(1, N_MODES + 1)
gamma = np.array([
    2.0 * quad(lambda x, m=m: f_np(x) * np.sin(m * np.pi * x), 0, 1, limit=400)[0]
    for m in n
])
beta = gamma / (n**2 * np.pi**2 - K**2)


def u_exact(x):
    return (beta[None, :] * np.sin(np.outer(x, n * np.pi))).sum(axis=1)


class PINN(nn.Module):
    def __init__(self, width=32, depth=3):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def second_derivative(u, x):
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    return torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]


net = PINN()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
x_b = torch.tensor([[0.0], [1.0]])

for it in range(N_ITER + 1):
    x = torch.rand(N_COLLOCATION, 1, requires_grad=True)
    u = net(x)
    u_xx = second_derivative(u, x)
    residual = u_xx + K**2 * u + f_torch(x)
    loss_pde = (residual**2).mean()
    loss_bc = (net(x_b)**2).mean()
    loss = loss_pde + MU * loss_bc

    opt.zero_grad()
    loss.backward()
    opt.step()

    if it % 1000 == 0:
        print(f"iter {it:5d}   loss {loss.item():.3e}   "
              f"(pde {loss_pde.item():.3e}, bc {loss_bc.item():.3e})")

x_fixed = torch.rand(N_COLLOCATION, 1, requires_grad=True)
lbfgs = torch.optim.LBFGS(net.parameters(), max_iter=500, history_size=50,
                          tolerance_grad=1e-12, tolerance_change=1e-14)


def closure():
    lbfgs.zero_grad()
    u = net(x_fixed)
    u_xx = second_derivative(u, x_fixed)
    r = u_xx + K**2 * u + f_torch(x_fixed)
    l = (r**2).mean() + MU * (net(x_b)**2).mean()
    l.backward()
    return l


lbfgs.step(closure)
print(f"after L-BFGS   loss {closure().item():.3e}")

xs = np.linspace(0, 1, 400)
u_ex = u_exact(xs)
with torch.no_grad():
    u_nn = net(torch.tensor(xs, dtype=torch.float32).reshape(-1, 1)).numpy().ravel()

rel_err = np.linalg.norm(u_ex - u_nn) / np.linalg.norm(u_ex)
print(f"\nmax |u_exact| = {np.abs(u_ex).max():.5f}")
print(f"relative L2 error = {rel_err:.3%}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(xs, u_ex, "k-", lw=2.5, label="exact (Fourier)")
ax[0].plot(xs, u_nn, "r--", lw=2, label="PINN")
ax[0].set_xlabel("x")
ax[0].set_ylabel("u(x)")
ax[0].set_title(f"k = {K},  f(x) = exp(-x^2/20)")
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot(xs, u_ex - u_nn, "b-")
ax[1].set_xlabel("x")
ax[1].set_ylabel("u_exact - u_PINN")
ax[1].set_title(f"pointwise error (rel. L2 = {rel_err:.2%})")
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/pinn_vs_exact.png", dpi=140)
print("saved figures/pinn_vs_exact.png")