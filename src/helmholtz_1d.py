import argparse
import csv
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from scipy.fft import dst
from scipy.integrate import quad
from scipy.stats import qmc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullLocator

torch.set_default_dtype(torch.float64)

KPI_SWEEP = [6.5, 6.25, 6.125, 6.0625, 6.03125]
SAMPLER = "grid"
N_COLLOCATION = 100
RESAMPLE = False
MU = 1000.0
HARD_BC = False
STRETCH = False
N_ITER = 15000
LBFGS_ITER = 5000
LR = 1e-3
WIDTH = 32
DEPTH = 3
N_MODES = 1000
SEED = 5

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
VIOLET = "#4a3aa7"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.titlecolor": INK, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})


def f_np(x):
    return np.exp(-20.0 * x**2)


def f_torch(x):
    return torch.exp(-20.0 * x**2)


n = np.arange(1, N_MODES + 1)
gamma = np.array([
    2.0 * quad(lambda x, m=m: f_np(x) * np.sin(m * np.pi * x), 0, 1, limit=400)[0]
    for m in n
])


def exact_coefficients(k):
    return gamma / (n**2 * np.pi**2 - k**2)


def u_exact(x, k):
    return (exact_coefficients(k)[None, :] * np.sin(np.outer(x, n * np.pi))).sum(axis=1)


def nearest_resonance(k):
    gaps = np.abs(n**2 * np.pi**2 - k**2)
    i = gaps.argmin()
    return int(n[i]), gaps[i]


def sine_coefficients(values):
    return dst(values[1:-1], type=1) / (len(values) - 1)


def l2(values):
    return np.sqrt(np.mean(values**2))


def config_tag(mu, hard_bc, stretch):
    bc = "hard" if hard_bc else f"mu{mu:g}"
    return f"{bc}_{'stretch' if stretch else 'plain'}"


def config_label(mu, hard_bc, stretch, ascii=False):
    m = "mu" if ascii else "μ"
    bc = "hard BC" if hard_bc else f"weak BC {m} = {mu:g}"
    return f"{bc}, {'input stretch' if stretch else 'no scaling'}"


def collocation(n_pts, sampler, seed):
    if sampler == "grid":
        pts = np.linspace(0.0, 1.0, n_pts + 2)[1:-1]
    elif sampler == "random":
        pts = np.random.rand(n_pts)
    elif sampler == "sobol":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pts = qmc.Sobol(d=1, scramble=True, seed=seed).random(n_pts).ravel()
    elif sampler == "halton":
        pts = qmc.Halton(d=1, scramble=True, seed=seed).random(n_pts).ravel()
    else:
        raise ValueError(sampler)
    t = torch.tensor(pts, dtype=torch.get_default_dtype()).reshape(-1, 1)
    return t.requires_grad_(True)


class PINN(nn.Module):
    def __init__(self, k, hard_bc=HARD_BC, stretch=STRETCH, width=WIDTH, depth=DEPTH):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        self.hard_bc = hard_bc
        self.stretch = stretch
        self.s_in = max(1.0, k / (2 * np.pi))

    def forward(self, x):
        z = self.s_in * (2.0 * x - 1.0) if self.stretch else x
        u = self.net(z)
        if self.hard_bc:
            return x * (1.0 - x) * u
        return u


def second_derivative(u, x):
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    return torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]


def run(k, n_coll=N_COLLOCATION, sampler=SAMPLER, resample=RESAMPLE, mu=MU,
        n_iter=N_ITER, lbfgs_iter=LBFGS_ITER, seed=SEED, hard_bc=HARD_BC, stretch=STRETCH,
        verbose=True, plot=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    net = PINN(k, hard_bc=hard_bc, stretch=stretch)
    x_b = torch.tensor([[0.0], [1.0]])
    x_coll = collocation(n_coll, sampler, seed)
    n_star, gap = nearest_resonance(k)

    def residual(x):
        u = net(x)
        return second_derivative(u, x) + k**2 * u + f_torch(x)

    def loss_parts(x):
        return (residual(x)**2).mean(), (net(x_b)**2).mean()

    if verbose:
        print("=" * 78)
        print(f"k = {k:.6f}   k/pi = {k / np.pi:g}   nearest resonance n* = {n_star}   "
              f"gap |n*^2 pi^2 - k^2| = {gap:.4f}")
        print(f"{config_label(mu, hard_bc, stretch, ascii=True)}   sampler = {sampler}   "
              f"n_coll = {n_coll}   n_iter = {n_iter}   lbfgs_iter = {lbfgs_iter}\n")

    opt = torch.optim.Adam(net.parameters(), lr=LR)
    hist_adam = []
    for it in range(n_iter + 1):
        if resample:
            x_coll = collocation(n_coll, sampler, seed + it)

        loss_pde, loss_bc = loss_parts(x_coll)
        loss = loss_pde + mu * loss_bc
        hist_adam.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

        if verbose and it % 1000 == 0:
            print(f"iter {it:5d}   loss {loss.item():.3e}   "
                  f"(pde {loss_pde.item():.3e}, bc {loss_bc.item():.3e})")

    pde_adam, bc_adam = (t.item() for t in loss_parts(x_coll))
    loss_adam = pde_adam + mu * bc_adam

    lbfgs = torch.optim.LBFGS(net.parameters(), max_iter=lbfgs_iter, history_size=50,
                              tolerance_grad=1e-12, tolerance_change=1e-14,
                              line_search_fn="strong_wolfe")
    hist_lbfgs = []

    def closure():
        lbfgs.zero_grad()
        loss_pde, loss_bc = loss_parts(x_coll)
        l = loss_pde + mu * loss_bc
        l.backward()
        hist_lbfgs.append(l.item())
        return l

    if lbfgs_iter > 0:
        lbfgs.step(closure)
    pde_final, bc_final = (t.item() for t in loss_parts(x_coll))
    loss_final = pde_final + mu * bc_final

    xs = np.linspace(0.0, 1.0, 2001)
    u_ex = u_exact(xs, k)
    xt = torch.tensor(xs, dtype=torch.get_default_dtype()).reshape(-1, 1).requires_grad_(True)
    res_dense = residual(xt).abs().max().item()
    with torch.no_grad():
        u_nn = net(xt).numpy().ravel()

    err = u_ex - u_nn
    u_l2 = l2(u_ex)
    u_max = np.abs(u_ex).max()
    beta_star = abs(exact_coefficients(k)[n_star - 1])
    err_l2 = l2(err)
    err_max = np.abs(err).max()
    rel_err = err_l2 / u_l2

    ex_hat = sine_coefficients(u_ex)
    err_hat = sine_coefficients(err)
    ref = np.linalg.norm(ex_hat)
    rel_spec = np.abs(err_hat) / ref
    ex_spec = np.abs(ex_hat) / ref
    spec_shape = np.abs(err_hat) / np.linalg.norm(err_hat)
    energy = err_hat**2 / (err_hat**2).sum()
    share_star = energy[n_star - 1]
    top = np.argsort(energy)[::-1][:3] + 1

    if verbose:
        print(f"\nafter Adam     loss {loss_adam:.3e}   (pde {pde_adam:.3e}, bc {bc_adam:.3e})")
        print(f"after L-BFGS   loss {loss_final:.3e}   (pde {pde_final:.3e}, bc {bc_final:.3e})"
              f"   [{len(hist_lbfgs)} evaluations]")
        print(f"\nsize of u        L2 = {u_l2:.3e}   max = {u_max:.3e}   "
              f"|beta_{n_star}| = {beta_star:.3e}")
        print(f"error            L2 = {err_l2:.3e}   max = {err_max:.3e}   "
              f"relative = {rel_err:.3e}")
        print(f"error spectrum   share in mode {n_star} = {share_star:.1%}   largest: "
              + "   ".join(f"n={m} ({energy[m - 1]:.1%})" for m in top))
        print(f"max |residual| on dense grid = {res_dense:.3e}")

    if plot:
        os.makedirs("figures", exist_ok=True)
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))

        a = ax[0, 0]
        a.plot(xs, u_ex, color=INK, lw=2, label="exact (Fourier)")
        a.plot(xs, u_nn, color=BLUE, lw=2, ls="--", label="PINN")
        a.set_xlabel("x")
        a.set_ylabel("u(x)")
        a.set_title(f"k = {k / np.pi:g}π   ({config_label(mu, hard_bc, stretch)})")
        a.legend(loc="best")

        a = ax[0, 1]
        a.plot(xs, err, color=BLUE, lw=1.5)
        a.axhline(0, color=MUTED, lw=0.6)
        a.set_xlabel("x")
        a.set_ylabel("u_exact - u_PINN")
        a.set_title(f"pointwise error,  rel. L2 = {rel_err:.2e}")

        a = ax[1, 0]
        n_max = max(40, int(3 * k / np.pi))
        modes = np.arange(1, n_max + 1)
        a.semilogy(modes, ex_spec[:n_max], "o", color=MUTED, ms=5, mfc="none",
                   label="exact  |ûₙ| / ‖û‖")
        a.semilogy(modes, rel_spec[:n_max], "o-", color=BLUE, ms=4, lw=1,
                   label="error  |êₙ| / ‖û‖")
        a.axvline(k / np.pi, color=ORANGE, lw=1, ls="--", label=f"k/π = {k / np.pi:g}")
        a.set_xlabel("sine mode n")
        a.set_ylabel("coefficient / ‖û‖")
        title = f"error spectrum (sine FFT),  n*={n_star}: {share_star:.0%}"
        if top[0] != n_star:
            title += f",  top n={top[0]}: {energy[top[0] - 1]:.0%}"
        a.set_title(title)
        a.legend(loc="upper right")

        a = ax[1, 1]
        a.semilogy(np.arange(len(hist_adam)), hist_adam, color=BLUE, lw=0.8,
                   label="Adam (per iteration)")
        if hist_lbfgs:
            a.semilogy(len(hist_adam) + np.arange(len(hist_lbfgs)), hist_lbfgs,
                       color=ORANGE, lw=1.2, label="L-BFGS (per evaluation)")
        a.set_xlabel("step")
        a.set_ylabel("loss")
        a.set_title(f"loss:  after Adam {loss_adam:.2e},  after L-BFGS {loss_final:.2e}")
        a.legend(loc="upper right")

        for a in ax.ravel():
            a.grid(color=GRID, lw=0.6)
            a.set_axisbelow(True)

        plt.tight_layout()
        path = f"figures/pinn_kpi{k / np.pi:g}_{config_tag(mu, hard_bc, stretch)}.png"
        plt.savefig(path, dpi=140)
        plt.close(fig)
        if verbose:
            print(f"saved {path}")

    return dict(kpi=k / np.pi, k=k, n_star=n_star, gap=gap, mu=mu, hard_bc=hard_bc,
                stretch=stretch, n_coll=n_coll, u_l2=u_l2, u_max=u_max, beta_star=beta_star,
                err_l2=err_l2, err_max=err_max, rel_err=rel_err, share_star=share_star,
                loss_adam=loss_adam, pde_adam=pde_adam, bc_adam=bc_adam,
                loss_final=loss_final, pde_final=pde_final, bc_final=bc_final,
                max_res=res_dense, lbfgs_evals=len(hist_lbfgs), spec_shape=spec_shape)


CSV_KEYS = ["kpi", "k", "n_star", "gap", "mu", "hard_bc", "stretch", "n_coll", "u_l2",
            "u_max", "beta_star", "err_l2", "err_max", "rel_err", "share_star", "loss_adam",
            "pde_adam", "bc_adam", "loss_final", "pde_final", "bc_final", "max_res",
            "lbfgs_evals"]


def summarize(rows):
    mu, hard_bc, stretch = rows[0]["mu"], rows[0]["hard_bc"], rows[0]["stretch"]
    kpis = [r["kpi"] for r in rows]
    stem = f"resonance_kpi{min(kpis):g}-{max(kpis):g}_{config_tag(mu, hard_bc, stretch)}"

    os.makedirs("results", exist_ok=True)
    with open(f"results/{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_KEYS)
        for r in rows:
            w.writerow([r[key] for key in CSV_KEYS])

    ns = rows[0]["n_star"]
    print("\n" + "=" * 78)
    print(f"SUMMARY   {config_label(mu, hard_bc, stretch, ascii=True)}")
    print(f"{'k/pi':>9} {'gap':>8} {'||u||':>10} {'|beta_' + str(ns) + '|':>10} "
          f"{'||err||':>10} {'rel_err':>10} {'share_' + str(ns):>9} "
          f"{'loss_Adam':>10} {'loss_LBFGS':>10}")
    for r in rows:
        print(f"{r['kpi']:>9g} {r['gap']:>8.2f} {r['u_l2']:>10.3e} {r['beta_star']:>10.3e} "
              f"{r['err_l2']:>10.3e} {r['rel_err']:>10.3e} {r['share_star']:>9.1%} "
              f"{r['loss_adam']:>10.3e} {r['loss_final']:>10.3e}")
    print(f"\nsaved results/{stem}.csv")

    if len(rows) < 2:
        return

    rows = sorted(rows, key=lambda r: -r["gap"])
    d = np.array([abs(r["kpi"] - r["n_star"]) for r in rows])
    gaps = np.array([r["gap"] for r in rows])

    def col(key):
        return np.array([r[key] for r in rows])

    def guide(a, y0):
        a.loglog(d, y0 * gaps[0] / gaps, ":", color=MUTED, lw=1.2,
                 label="×2 per halving (∝ 1/gap)")

    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))

    a = ax[0, 0]
    a.loglog(d, col("u_l2"), "o-", color=INK, lw=2, ms=7, label="‖u‖ (L2)")
    a.loglog(d, col("u_max"), "s--", color=INK, lw=1.5, ms=6, label="max |u|")
    a.loglog(d, col("beta_star"), "o-", color=VIOLET, lw=2, ms=7,
             label=f"|β_{ns}|  (mode-{ns} part of u)")
    guide(a, col("u_l2")[0])
    a.set_title("1.  Does u blow up?")
    a.set_ylabel("size of exact solution")

    a = ax[0, 1]
    a.loglog(d, col("err_l2"), "o-", color=BLUE, lw=2, ms=7, label="‖u − û‖ (L2)")
    a.loglog(d, col("err_max"), "s--", color=BLUE, lw=1.5, ms=6, label="max |u − û|")
    guide(a, col("err_l2")[0])
    a.set_title("2a.  Does the error blow up?  (absolute)")
    a.set_ylabel("error")

    a = ax[0, 2]
    a.loglog(d, col("rel_err"), "o-", color=BLUE, lw=2, ms=7, label="‖u − û‖ / ‖u‖")
    a.set_title("2b.  Does the error blow up?  (relative to u)")
    a.set_ylabel("relative error")

    a = ax[1, 0]
    n_max = max(20, 3 * ns)
    modes = np.arange(1, n_max + 1)
    colors = RAMP if len(rows) <= len(RAMP) else plt.cm.Blues(np.linspace(0.4, 1.0, len(rows)))
    for r, c in zip(rows, colors):
        a.semilogy(modes, r["spec_shape"][:n_max], "o-", color=c, lw=1.5, ms=4,
                   label=f"k = {r['kpi']:g}π")
    a.axvline(ns, color=MUTED, lw=0.8, ls="--")
    a.xaxis.set_major_locator(MaxNLocator(integer=True))
    a.set_xlabel("sine mode n")
    a.set_ylabel("|êₙ| / ‖ê‖")
    a.set_title("3a.  Does the error spectrum get peakier?")

    a = ax[1, 1]
    a.semilogx(d, 100 * col("share_star"), "o-", color=VIOLET, lw=2, ms=7,
               label=f"share of error in mode {ns}")
    a.set_ylim(0, 100)
    a.set_ylabel("% of error energy")
    a.set_title(f"3b.  Share of the error in mode {ns}")

    a = ax[1, 2]
    a.loglog(d, col("loss_adam"), "o-", color=BLUE, lw=2, ms=7, label="after Adam")
    a.loglog(d, col("loss_final"), "o-", color=ORANGE, lw=2, ms=7, label="after L-BFGS")
    a.set_title("4.  Does the loss get larger?")
    a.set_ylabel("loss  (pde)" if hard_bc else "loss  (pde + μ · bc)")

    for a in ax.ravel():
        if a.get_yscale() == "log":
            lo, hi = a.get_ylim()
            subs = (1.0, 2.0, 5.0) if np.log10(hi / lo) < 3 else (1.0,)
            a.yaxis.set_major_locator(LogLocator(base=10, subs=subs))
            a.yaxis.set_minor_locator(NullLocator())
            a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        a.grid(color=GRID, lw=0.6, which="major")
        a.set_axisbelow(True)
        if len(a.get_legend_handles_labels()[0]) > 1:
            a.legend(loc="best")
    for a in [ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 1], ax[1, 2]]:
        a.set_xticks(d)
        a.set_xticklabels([f"{r['kpi']:g}" for r in rows])
        a.xaxis.set_minor_locator(NullLocator())
        a.invert_xaxis()
        a.set_xlabel(f"k / π     (log scale, closer to {ns} →)")

    fig.suptitle(f"Approaching resonance k → {ns}π     "
                 f"({config_label(mu, hard_bc, stretch)},  n_coll = {rows[0]['n_coll']})",
                 fontsize=13)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    path = f"figures/{stem}.png"
    plt.savefig(path, dpi=140)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kpi", type=float, nargs="+", default=KPI_SWEEP)
    p.add_argument("--n-coll", type=int, default=N_COLLOCATION)
    p.add_argument("--sampler", default=SAMPLER)
    p.add_argument("--resample", action="store_true", default=RESAMPLE)
    p.add_argument("--mu", type=float, default=MU)
    p.add_argument("--hard-bc", action="store_true", default=HARD_BC)
    p.add_argument("--stretch", action="store_true", default=STRETCH)
    p.add_argument("--n-iter", type=int, default=N_ITER)
    p.add_argument("--lbfgs-iter", type=int, default=LBFGS_ITER)
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args()
    rows = [run(kpi * np.pi, a.n_coll, a.sampler, a.resample, a.mu, a.n_iter,
                a.lbfgs_iter, a.seed, hard_bc=a.hard_bc, stretch=a.stretch)
            for kpi in a.kpi]
    summarize(rows)