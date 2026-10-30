import argparse
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, grad, jacrev, vmap
from scipy.fft import dst
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullLocator

torch.set_default_dtype(torch.float64)


K_MIN = 4.0
K_MAX = 6.0
K_TEST = None
N_COLLOCATION = 1000
N_BOUNDARY = 50
FOCUS = 0.0
FOCUS_WIDTH = 0.1
SCALE_K = True
OPTIMIZER = "lm"
MU = "auto"
MU_EVERY = {"lm": 10, "adam": 100}
HARD_BC = False
LM_ITER = 3000
N_ITER = 20000
LBFGS_ITER = 5000
LR = 1e-3
WIDTH = 32
DEPTH = 3
N_MODES = 1000
N_FINE = 2**17
N_K_CURVE = 100
SEED = 5

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
VIOLET = "#4a3aa7"

plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.titlecolor": INK, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})


def f_np(x):
    return np.exp(-20.0 * x**2)


def f_torch(x):
    return torch.exp(-20.0 * x**2)


def sine_coefficients(values):
    return dst(values[1:-1], type=1) / (len(values) - 1)


n = np.arange(1, N_MODES + 1)
x_fine = np.linspace(0.0, 1.0, N_FINE + 1)
gamma = sine_coefficients(f_np(x_fine))[:N_MODES]


def forcing_np(x, k, forcing="real"):
    if forcing == "real":
        return f_np(x)
    if forcing == "imag":
        return 1j * f_np(x)
    if forcing == "harmonic":
        return f_np(x) * np.exp(1j * k * x)
    raise ValueError(forcing)


def forcing_coefficients(k, forcing="real"):
    if forcing == "real":
        return gamma
    if forcing == "imag":
        return 1j * gamma
    return sine_coefficients(forcing_np(x_fine, k, forcing))[:N_MODES]


def exact_coefficients(k, eta=0.0, forcing="real"):
    kappa2 = k**2 if eta == 0 else (k - 1j * eta)**2
    return forcing_coefficients(k, forcing) / (n**2 * np.pi**2 - kappa2)


def u_exact(x, k, eta=0.0, forcing="real"):
    return (exact_coefficients(k, eta, forcing)[None, :] * np.sin(np.outer(x, n * np.pi))).sum(axis=1)


def u_exact_many(x, ks, eta=0.0, forcing="real"):
    coeffs = np.stack([exact_coefficients(k, eta, forcing) for k in ks], axis=1)
    return np.sin(np.outer(x, n * np.pi)) @ coeffs


def nearest_resonance(k):
    gaps = np.abs(n**2 * np.pi**2 - k**2)
    i = gaps.argmin()
    return int(n[i]), gaps[i]


def resonances_in(k_min, k_max):
    return [int(m) for m in n if k_min - 1e-9 <= m * np.pi <= k_max + 1e-9]


def l2(values, axis=None):
    return np.sqrt(np.mean(np.abs(values)**2, axis=axis))


def config_tag(mu, hard_bc, scale_k, optimizer, mu_auto):
    bc = "hard" if hard_bc else ("muauto" if mu_auto else f"mu{mu:g}")
    return f"{bc}_{'kscaled' if scale_k else 'kraw'}_{optimizer}"


def config_label(mu, hard_bc, scale_k, optimizer, mu_auto, ascii=False):
    m = "mu" if ascii else "μ"
    if hard_bc:
        bc = "hard BC"
    else:
        bc = f"weak BC {m} = " + ("auto" if mu_auto else f"{mu:g}")
    kin = "k scaled to [-1, 1]" if scale_k else "raw k"
    opt = "LM" if optimizer == "lm" else "Adam + L-BFGS"
    return f"{bc}, {kin}, {opt}"


def near_resonance_k(n_pts, k_min, k_max, width):
    windows = [(max(k_min, m * np.pi - width), min(k_max, m * np.pi + width))
               for m in resonances_in(k_min - width, k_max + width)]
    windows = [(a, b) for a, b in windows if b > a]
    if not windows:
        return k_min + (k_max - k_min) * np.random.rand(n_pts)
    starts = np.array([a for a, _ in windows])
    edges = np.r_[0.0, np.cumsum([b - a for a, b in windows])]
    s = edges[-1] * np.random.rand(n_pts)
    i = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(windows) - 1)
    return starts[i] + s - edges[i]


def sample_points(n_coll, n_bc, k_min, k_max, focus=FOCUS, focus_width=FOCUS_WIDTH * np.pi):
    n_focus = int(round(focus * n_coll))
    x = np.random.rand(n_coll)
    k = k_min + (k_max - k_min) * np.random.rand(n_coll - n_focus)
    if n_focus > 0:
        k = np.r_[k, near_resonance_k(n_focus, k_min, k_max, focus_width)]
    kb = k_min + (k_max - k_min) * np.random.rand(n_bc)
    X = torch.tensor(np.stack([x, k], axis=1))
    Xb = torch.tensor(np.stack([np.r_[np.zeros(n_bc), np.ones(n_bc)], np.r_[kb, kb]], axis=1))
    return X, Xb


def pairs(x, k):
    return torch.tensor(np.stack([x, np.full_like(x, k)], axis=1))


def on_grid(net, x, ks):
    X = torch.tensor(np.stack([np.tile(x, len(ks)), np.repeat(ks, len(x))], axis=1))
    with torch.no_grad():
        return net(X).numpy().reshape(len(ks), len(x)).T


def close_to_resonances(k_min, k_max):
    d = np.pi * np.logspace(-6, -2, 15)
    ks = np.array([m * np.pi + s * d for m in resonances_in(k_min, k_max) for s in (-1, 1)]).ravel()
    return ks[(ks > k_min) & (ks < k_max)]


class PINN(nn.Module):
    def __init__(self, k_min, k_max, hard_bc=HARD_BC, scale_k=SCALE_K, width=WIDTH, depth=DEPTH):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        self.hard_bc = hard_bc
        self.scale_k = scale_k
        self.k_mid = 0.5 * (k_min + k_max)
        self.k_half = max(0.5 * (k_max - k_min), 1e-12)

    def forward(self, X):
        x, k = X[:, 0:1], X[:, 1:2]
        k_in = (k - self.k_mid) / self.k_half if self.scale_k else k
        u = self.net(torch.cat([x, k_in], dim=1))
        if self.hard_bc:
            return x * (1.0 - x) * u
        return u


def residual(net, X):
    X = X.detach().requires_grad_(True)
    u = net(X)
    u_x = torch.autograd.grad(u, X, torch.ones_like(u), create_graph=True)[0][:, 0:1]
    u_xx = torch.autograd.grad(u_x, X, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    x, k = X[:, 0:1], X[:, 1:2]
    return u_xx + k**2 * u + f_torch(x)


def flat_params(net):
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()])


def functional_parts(net, X, Xb):
    names = [nm for nm, _ in net.named_parameters()]
    shapes = [p.shape for _, p in net.named_parameters()]
    sizes = [p.numel() for _, p in net.named_parameters()]

    def unflat(t):
        out, i = {}, 0
        for nm, sh, sz in zip(names, shapes, sizes):
            out[nm] = t[i:i + sz].reshape(sh)
            i += sz
        return out

    def u_point(t, p):
        return functional_call(net, unflat(t), (p.reshape(1, 2),)).reshape(())

    def u_x(t, p):
        return grad(u_point, argnums=1)(t, p)[0]

    def r_point(t, p):
        return grad(u_x, argnums=1)(t, p)[0] + p[1]**2 * u_point(t, p) + f_torch(p[0])

    X, Xb = X.detach(), Xb.detach()
    r_all = vmap(r_point, in_dims=(None, 0))
    u_all = vmap(u_point, in_dims=(None, 0))
    jr_all = vmap(jacrev(r_point), in_dims=(None, 0))
    ju_all = vmap(jacrev(u_point), in_dims=(None, 0))
    return (lambda t: r_all(t, X), lambda t: u_all(t, Xb),
            lambda t: jr_all(t, X), lambda t: ju_all(t, Xb))


def ntk_mu(j_pde, j_bc):
    return ((j_pde**2).sum(1).mean() / (j_bc**2).sum(1).mean()).item()


def train_lm(net, X, Xb, mu, mu_auto, hard_bc, lm_iter, verbose):
    r_fn, b_fn, jr_fn, jb_fn = functional_parts(net, X, Xb)
    n_c, n_b = X.shape[0], Xb.shape[0]
    auto = mu_auto and not hard_bc
    theta = flat_params(net)

    def residuals(t, m):
        r = r_fn(t) / math.sqrt(n_c)
        if hard_bc:
            return r
        return torch.cat([r, b_fn(t) * math.sqrt(m / n_b)])

    rv, loss, lam = None, None, None
    t0 = time.time()
    it = 0
    for it in range(1, lm_iter + 1):
        j_pde = jr_fn(theta)
        j_bc = None if hard_bc else jb_fn(theta)
        if auto and (it == 1 or it % MU_EVERY["lm"] == 0):
            mu = ntk_mu(j_pde, j_bc)
            rv = None
        if rv is None:
            rv = residuals(theta, mu)
            loss = (rv**2).sum().item()
        if hard_bc:
            jac = j_pde / math.sqrt(n_c)
        else:
            jac = torch.cat([j_pde / math.sqrt(n_c), j_bc * math.sqrt(mu / n_b)])
        dual = jac.shape[0] <= jac.shape[1]
        gram = jac @ jac.T if dual else jac.T @ jac
        rhs = rv if dual else jac.T @ rv
        eye = torch.eye(gram.shape[0], dtype=gram.dtype)
        if lam is None:
            lam = 1e-3 * (jac**2).sum().item() / jac.shape[0]
        accepted = False
        for _ in range(60):
            step = torch.linalg.solve(gram + lam * eye, rhs)
            trial = theta - (jac.T @ step if dual else step)
            rv_trial = residuals(trial, mu)
            loss_trial = (rv_trial**2).sum().item()
            if loss_trial < loss:
                theta, rv, loss = trial, rv_trial, loss_trial
                lam = max(lam / 3, 1e-14)
                accepted = True
                break
            lam *= 2
        if verbose and it % 500 == 0:
            pde = (rv[:n_c]**2).sum().item()
            bc = 0.0 if hard_bc else (rv[n_c:]**2).sum().item() / mu
            print(f"LM iter {it:5d}   loss {loss:.3e}   (pde {pde:.3e}, bc {bc:.3e})"
                  + ("" if hard_bc else f"   mu {mu:.3e}") + f"   {time.time() - t0:.0f}s")
        if not accepted or loss < 1e-24:
            break

    torch.nn.utils.vector_to_parameters(theta, net.parameters())
    return mu, it


def run(k_min=K_MIN * np.pi, k_max=K_MAX * np.pi, k_test=K_TEST, n_coll=N_COLLOCATION,
        n_bc=N_BOUNDARY, focus=FOCUS, focus_width=FOCUS_WIDTH * np.pi, mu=MU, scale_k=SCALE_K,
        hard_bc=HARD_BC, optimizer=OPTIMIZER, lm_iter=LM_ITER, n_iter=N_ITER,
        lbfgs_iter=LBFGS_ITER, width=WIDTH, depth=DEPTH, seed=SEED, verbose=True, plot=True):
    mu_auto = mu == "auto"
    mu = 0.0 if mu_auto else float(mu)
    torch.manual_seed(seed)
    np.random.seed(seed)

    net = PINN(k_min, k_max, hard_bc=hard_bc, scale_k=scale_k, width=width, depth=depth)
    X, Xb = sample_points(n_coll, n_bc, k_min, k_max, focus, focus_width)
    test_kind = "random" if k_test is None else "chosen"
    if k_test is None:
        k_test = np.random.default_rng().uniform(k_min, k_max)
    res_in = resonances_in(k_min, k_max)
    points = f"n_coll = {n_coll}" + (f" ({focus:.0%} within {focus_width / np.pi:g}pi of a resonance)"
                                      if focus > 0 else "")

    if verbose:
        print("=" * 78)
        print(f"training range   {k_min / np.pi:g}pi <= k <= {k_max / np.pi:g}pi   resonances in range: "
              + (", ".join(f"{m}pi" for m in res_in) if res_in else "none"))
        steps = (f"lm_iter = {lm_iter}" if optimizer == "lm"
                 else f"n_iter = {n_iter}   lbfgs_iter = {lbfgs_iter}")
        print(f"{config_label(mu, hard_bc, scale_k, optimizer, mu_auto, ascii=True)}   "
              f"{points}   n_bc = {n_bc}   width = {width}   depth = {depth}   {steps}\n")

    hist_lbfgs = []
    lm_iters = 0

    def loss_parts():
        return (residual(net, X)**2).mean(), (net(Xb)**2).mean()

    if optimizer == "lm":
        mu, lm_iters = train_lm(net, X, Xb, mu, mu_auto, hard_bc, lm_iter, verbose)
    else:
        auto = mu_auto and not hard_bc
        jr_fn, jb_fn = functional_parts(net, X, Xb)[2:]
        if auto:
            theta = flat_params(net)
            mu = ntk_mu(jr_fn(theta), jb_fn(theta))
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        for it in range(n_iter + 1):
            if auto and it > 0 and it % MU_EVERY["adam"] == 0:
                theta = flat_params(net)
                mu = ntk_mu(jr_fn(theta), jb_fn(theta))

            loss_pde, loss_bc = loss_parts()
            loss = loss_pde + mu * loss_bc

            opt.zero_grad()
            loss.backward()
            opt.step()

            if verbose and it % 1000 == 0:
                print(f"iter {it:5d}   loss {loss.item():.3e}   "
                      f"(pde {loss_pde.item():.3e}, bc {loss_bc.item():.3e})"
                      + (f"   mu {mu:.3e}" if auto else ""))

        lbfgs = torch.optim.LBFGS(net.parameters(), max_iter=lbfgs_iter, history_size=50,
                                  tolerance_grad=1e-12, tolerance_change=1e-14,
                                  line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            loss_pde, loss_bc = loss_parts()
            l = loss_pde + mu * loss_bc
            l.backward()
            hist_lbfgs.append(l.item())
            return l

        if lbfgs_iter > 0:
            lbfgs.step(closure)

    pde_final, bc_final = (t.item() for t in loss_parts())
    loss_final = pde_final + mu * bc_final

    xs = np.linspace(0.0, 1.0, 2001)
    n_star, gap = nearest_resonance(k_test)
    u_ex = u_exact(xs, k_test)
    X_test = pairs(xs, k_test)
    res_test = residual(net, X_test).abs().max().item()
    with torch.no_grad():
        u_nn = net(X_test).numpy().ravel()

    err = u_ex - u_nn
    u_l2 = l2(u_ex)
    u_max = np.abs(u_ex).max()
    err_l2 = l2(err)
    err_max = np.abs(err).max()
    rel_err = err_l2 / u_l2

    ex_hat = sine_coefficients(u_ex)
    err_hat = sine_coefficients(err)
    ref = np.linalg.norm(ex_hat)
    rel_spec = np.abs(err_hat) / ref
    ex_spec = np.abs(ex_hat) / ref
    energy = err_hat**2 / (err_hat**2).sum()
    share_star = energy[n_star - 1]
    top = np.argsort(energy)[::-1][:3] + 1

    k_curve = k_min + (np.arange(N_K_CURVE) + 0.5) * (k_max - k_min) / N_K_CURVE
    U_ex = u_exact_many(xs, k_curve)
    U_nn = on_grid(net, xs, k_curve)
    curve_u = l2(U_ex, axis=0)
    curve_nn = l2(U_nn, axis=0)
    curve_err = l2(U_ex - U_nn, axis=0)
    curve_rel = curve_err / curve_u
    worst = curve_rel.argmax()
    k_norm = np.sort(np.r_[k_curve, close_to_resonances(k_min, k_max)])
    norm_u = l2(u_exact_many(xs, k_norm), axis=0)
    norm_nn = l2(on_grid(net, xs, k_norm), axis=0)

    if verbose:
        if optimizer == "lm":
            print(f"\nafter LM       loss {loss_final:.3e}   (pde {pde_final:.3e}, bc {bc_final:.3e})"
                  f"   [{lm_iters} iterations]")
        else:
            print(f"\nafter L-BFGS   loss {loss_final:.3e}   (pde {pde_final:.3e}, bc {bc_final:.3e})"
                  f"   [{len(hist_lbfgs)} evaluations]")
        if mu_auto and not hard_bc:
            k_mid = 0.5 * (k_min + k_max)
            print(f"automatic mu = {mu:.3e}   (= {mu / k_mid**4:.2f} k_mid^4)")
        print(f"\nTEST at k = {k_test / np.pi:.4f}pi   ({test_kind}, not a training point)   "
              f"nearest resonance n* = {n_star}   gap |n*^2 pi^2 - k^2| = {gap:.4f}")
        print(f"size of u        L2 = {u_l2:.3e}   max = {u_max:.3e}")
        print(f"error            L2 = {err_l2:.3e}   max = {err_max:.3e}   "
              f"relative = {rel_err:.3e}")
        print(f"error spectrum   share in mode {n_star} = {share_star:.1%}   largest: "
              + "   ".join(f"n={m} ({energy[m - 1]:.1%})" for m in top))
        print(f"max |residual| at test k = {res_test:.3e}")
        print(f"\nacross the range ({N_K_CURVE} k values)   relative error: median "
              f"{np.median(curve_rel):.3e}   worst {curve_rel[worst]:.3e} "
              f"at k = {k_curve[worst] / np.pi:.4f}pi")
        print(f"solution norm    exact max {norm_u.max():.3e} at k = {k_norm[norm_u.argmax()] / np.pi:.6f}pi"
              f"   PINN max {norm_nn.max():.3e} at k = {k_norm[norm_nn.argmax()] / np.pi:.4f}pi")

    tag = config_tag(mu, hard_bc, scale_k, optimizer, mu_auto)
    stem = (f"param_k{k_min / np.pi:g}-{k_max / np.pi:g}_{tag}"
            + (f"_n{n_coll}" if n_coll != N_COLLOCATION else "")
            + (f"_focus{focus:g}" if focus > 0 else ""))
    os.makedirs("results", exist_ok=True)
    with open(f"results/{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kpi", "k", "n_star", "gap", "u_l2", "nn_l2", "err_l2", "rel_err", "test"])
        for kc, ul, nl, el, rl in zip(k_curve, curve_u, curve_nn, curve_err, curve_rel):
            ns, gp = nearest_resonance(kc)
            w.writerow([kc / np.pi, kc, ns, gp, ul, nl, el, rl, 0])
        w.writerow([k_test / np.pi, k_test, n_star, gap, u_l2, l2(u_nn), err_l2, rel_err, 1])
    if verbose:
        print(f"\nsaved results/{stem}.csv")

    if plot:
        os.makedirs("figures", exist_ok=True)
        fig = plt.figure(figsize=(18, 9.5))
        gs = fig.add_gridspec(2, 6)
        ax = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6]),
              fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])]

        a = ax[0]
        a.plot(xs, u_ex, color=INK, lw=2, label="real (exact Fourier)")
        a.plot(xs, u_nn, color=BLUE, lw=2, ls="--", label="tested (PINN)")
        a.set_xlabel("x")
        a.set_ylabel("u(x)")
        a.set_title(f"test k = {k_test / np.pi:.4f}π   ({test_kind}, not a training point)")
        a.legend(loc="best")

        a = ax[1]
        a.plot(xs, err, color=BLUE, lw=1.5)
        a.axhline(0, color=MUTED, lw=0.6)
        a.set_xlabel("x")
        a.set_ylabel("u_exact - u_PINN")
        a.set_title(f"pointwise error at test k,  rel. L2 = {rel_err:.2e}")

        a = ax[2]
        n_max = max(40, int(3 * k_test / np.pi))
        modes = np.arange(1, n_max + 1)
        a.semilogy(modes, ex_spec[:n_max], "o", color=MUTED, ms=5, mfc="none",
                   label="exact  |ûₙ| / ‖û‖")
        a.semilogy(modes, rel_spec[:n_max], "o-", color=BLUE, ms=4, lw=1,
                   label="error  |êₙ| / ‖û‖")
        a.axvline(k_test / np.pi, color=ORANGE, lw=1, ls="--", label=f"k/π = {k_test / np.pi:.4f}")
        a.set_xlabel("sine mode n")
        a.set_ylabel("coefficient / ‖û‖")
        title = f"error spectrum at test k,  n*={n_star}: {share_star:.0%}"
        if top[0] != n_star:
            title += f",  top n={top[0]}: {energy[top[0] - 1]:.0%}"
        a.set_title(title)
        a.legend(loc="upper right")

        a = ax[3]
        a.semilogy(k_curve / np.pi, curve_rel, "o-", color=VIOLET, lw=1.5, ms=3)
        for m in res_in:
            a.axvline(m, color=MUTED, lw=0.8, ls="--")
        a.semilogy([k_test / np.pi], [rel_err], "o", color=ORANGE, ms=9)
        a.set_xlabel("k / π" + ("     (dashed: resonances)" if res_in else ""))
        a.set_ylabel("relative error  ‖u − û‖ / ‖u‖")
        a.set_title(f"error across the k range,  median {np.median(curve_rel):.2e}  "
                    f"(orange: test k)")
        lo, hi = a.get_ylim()
        decades = np.log10(hi / lo)
        if decades < 3:
            if decades < 1:
                a.yaxis.set_major_locator(MaxNLocator(nbins=5))
            else:
                a.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            a.yaxis.set_minor_locator(NullLocator())
            a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

        a = ax[4]
        a.semilogy(k_norm / np.pi, norm_u, "-", color=INK, lw=2, label="exact  ‖u(k)‖")
        a.semilogy(k_norm / np.pi, norm_nn, "--", color=BLUE, lw=2, label="PINN  ‖û(k)‖")
        for m in res_in:
            a.axvline(m, color=MUTED, lw=0.8, ls="--")
        a.set_xlabel("k / π" + ("     (dashed: resonances)" if res_in else ""))
        a.set_ylabel("L2 norm over x")
        a.set_title(f"solution size across k,  max ‖û‖ = {norm_nn.max():.2e},  "
                    f"max ‖u‖ = {norm_u.max():.2e}")
        a.legend(loc="best")

        for a in ax:
            a.grid(color=GRID, lw=0.6)
            a.set_axisbelow(True)

        fig.suptitle(f"NN(x, k) trained on {k_min / np.pi:g}π ≤ k ≤ {k_max / np.pi:g}π     "
                     f"({config_label(mu, hard_bc, scale_k, optimizer, mu_auto)},  "
                     f"{points.replace('pi', 'π')})", fontsize=13)
        plt.tight_layout()
        path = f"figures/{stem}_test{k_test / np.pi:.4f}.png"
        plt.savefig(path, dpi=140)
        plt.close(fig)
        if verbose:
            print(f"saved {path}")

    return dict(k_min=k_min, k_max=k_max, k_test=k_test, n_star=n_star, gap=gap, mu=mu,
                u_l2=u_l2, err_l2=err_l2, rel_err=rel_err, share_star=share_star,
                loss_final=loss_final, max_res=res_test, lm_iters=lm_iters,
                k_curve=k_curve, curve_rel=curve_rel, k_norm=k_norm, norm_u=norm_u, norm_nn=norm_nn)


def mu_arg(s):
    return s if s == "auto" else float(s)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k-min", type=float, default=K_MIN)
    p.add_argument("--k-max", type=float, default=K_MAX)
    p.add_argument("--k-test", type=float, default=K_TEST)
    p.add_argument("--n-coll", type=int, default=N_COLLOCATION)
    p.add_argument("--n-bc", type=int, default=N_BOUNDARY)
    p.add_argument("--focus", type=float, default=FOCUS)
    p.add_argument("--focus-width", type=float, default=FOCUS_WIDTH)
    p.add_argument("--raw-k", action="store_true")
    p.add_argument("--optimizer", choices=["lm", "adam"], default=OPTIMIZER)
    p.add_argument("--mu", type=mu_arg, default=MU)
    p.add_argument("--hard-bc", action="store_true", default=HARD_BC)
    p.add_argument("--lm-iter", type=int, default=LM_ITER)
    p.add_argument("--n-iter", type=int, default=N_ITER)
    p.add_argument("--lbfgs-iter", type=int, default=LBFGS_ITER)
    p.add_argument("--width", type=int, default=WIDTH)
    p.add_argument("--depth", type=int, default=DEPTH)
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args()
    run(k_min=a.k_min * np.pi, k_max=a.k_max * np.pi,
        k_test=None if a.k_test is None else a.k_test * np.pi, n_coll=a.n_coll, n_bc=a.n_bc,
        focus=a.focus, focus_width=a.focus_width * np.pi,
        mu=a.mu, scale_k=not a.raw_k, hard_bc=a.hard_bc, optimizer=a.optimizer,
        lm_iter=a.lm_iter, n_iter=a.n_iter, lbfgs_iter=a.lbfgs_iter, width=a.width,
        depth=a.depth, seed=a.seed)