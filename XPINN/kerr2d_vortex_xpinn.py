from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


DEFAULT_BETA = 1.0
DEFAULT_D = 4.0
DEFAULT_Q = 1
DEFAULT_L = 6.0

DEFAULT_EPOCHS = 5000
DEFAULT_COLLOCATION_POINTS = 768
DEFAULT_QUAD_POINTS = 1536
DEFAULT_WIDTH = 64
DEFAULT_DEPTH = 5
DEFAULT_LR = 1.0e-3
DEFAULT_SEED = 20260706

DEFAULT_DEVICE = "auto"
DEFAULT_DTYPE = "float64"
DEFAULT_PRINT_EVERY = 100
DEFAULT_PLOT_DPI = 300

DEFAULT_LOSS_PDE_WEIGHT = 1.0
DEFAULT_LOSS_ROBIN_WEIGHT = 10.0
DEFAULT_LOSS_SCALE_WEIGHT = 0.0
DEFAULT_GRAD_CLIP = 10.0
DEFAULT_XPINN_SUBDOMAINS = 4
DEFAULT_XPINN_INTERFACE_WEIGHT = 10.0

DEFAULT_TOPOLOGY_RADIUS_FRACTIONS = (0.25, 0.50, 0.75, 0.95)
DEFAULT_TOPOLOGY_SAMPLES = 720
DEFAULT_RADIAL_PROFILE_POINTS = 600


FIELD_SPECS = (
    ("abs_phi", "Kerr2D_abs_phi.csv", "|phi(x,y)|", "standard"),
    ("abs_phi_squared", "Kerr2D_abs_phi_squared.csv", "|phi(x,y)|^2", "standard"),
    ("arg_phi", "Kerr2D_arg_phi.csv", "arg phi(x,y)", "phase"),
    ("e_nl", "Kerr2D_e_nl.csv", "e_nl(x,y)", "standard"),
    ("h_density", "Kerr2D_h_density.csv", "h(x,y)", "standard"),
    ("n_density", "Kerr2D_n_density.csv", "n(x,y)=|phi|^2", "standard"),
    ("k_density", "Kerr2D_k_density.csv", "k(x,y)=D|grad phi|^2", "standard"),
    ("m_density", "Kerr2D_m_density.csv", "m(x,y)=|phi|^4", "standard"),
    (
        "identity_density_eta",
        "Kerr2D_identity_density_eta.csv",
        "|m-k/2-beta n|",
        "standard",
    ),
)


@dataclass
class TrainResult:
    model: nn.Module
    history: np.ndarray
    total_training_seconds: float
    epoch_seconds: np.ndarray
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    final_total_loss: float


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int) -> None:
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(depth):
            layer = nn.Linear(last, width)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.extend((layer, nn.Tanh()))
            last = width
        final = nn.Linear(last, out_dim)
        nn.init.xavier_uniform_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RadialVortexPINN(nn.Module):
    def __init__(
        self,
        r_max: float,
        beta: float,
        dcoef: float,
        width: int,
        depth: int,
        subdomains: int,
    ) -> None:
        super().__init__()
        self.r_max = float(r_max)
        self.subdomains = int(subdomains)
        self.nets = nn.ModuleList(
            [MLP(1, 1, width, depth) for _ in range(self.subdomains)]
        )
        self.register_buffer("edges", torch.linspace(0.0, self.r_max, self.subdomains + 1))

    def local_coordinate(self, r: torch.Tensor, index: int) -> torch.Tensor:
        left = self.edges[index]
        right = self.edges[index + 1]
        return 2.0 * (r - left) / (right - left) - 1.0

    def forward_subdomain(self, r: torch.Tensor, index: int) -> torch.Tensor:
        return self.nets[index](self.local_coordinate(r, index))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        bins = torch.bucketize(r.squeeze(1), self.edges[1:-1], right=False)
        value = torch.zeros_like(r)
        for i in range(self.subdomains):
            mask = (bins == i).reshape(-1, 1).to(dtype=r.dtype, device=r.device)
            value = value + mask * self.forward_subdomain(r, i)
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XPINN for the 2D stationary Kerr vortex soliton."
    )
    parser.add_argument("--fdm-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--D", type=float, default=DEFAULT_D)
    parser.add_argument("--q", type=int, default=DEFAULT_Q)
    parser.add_argument("--L", type=float, default=DEFAULT_L)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--collocation", type=int, default=DEFAULT_COLLOCATION_POINTS)
    parser.add_argument("--quad-points", type=int, default=DEFAULT_QUAD_POINTS)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default=DEFAULT_DEVICE
    )
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), default=DEFAULT_DTYPE
    )
    parser.add_argument("--print-every", type=int, default=DEFAULT_PRINT_EVERY)
    parser.add_argument("--plot-dpi", type=int, default=DEFAULT_PLOT_DPI)
    parser.add_argument("--loss-pde-weight", type=float, default=DEFAULT_LOSS_PDE_WEIGHT)
    parser.add_argument(
        "--loss-robin-weight", type=float, default=DEFAULT_LOSS_ROBIN_WEIGHT
    )
    parser.add_argument(
        "--loss-scale-weight", type=float, default=DEFAULT_LOSS_SCALE_WEIGHT
    )
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--xpinn-subdomains", type=int, default=DEFAULT_XPINN_SUBDOMAINS)
    parser.add_argument(
        "--xpinn-interface-weight", type=float, default=DEFAULT_XPINN_INTERFACE_WEIGHT
    )
    return parser.parse_args()


def infer_fdm_dir(script_path: Path) -> Path:
    project_root = script_path.resolve().parent.parent
    for child in project_root.iterdir():
        if child.is_dir() and (child / "Kerr2D_abs_phi.csv").exists():
            return child
    raise FileNotFoundError(
        "Could not infer the FDM directory. Pass --fdm-dir with the folder "
        "that contains Kerr2D_abs_phi.csv."
    )


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if name == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def gradients(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def radial_quantities(
    model: nn.Module,
    r: torch.Tensor,
    beta: float,
    dcoef: float,
    q: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    f = model(r)
    fr = gradients(f, r)
    rr = r.squeeze(1)
    ff = f.squeeze(1)
    ffr = fr.squeeze(1)
    angular = (q * ff / rr) ** 2
    n_integrand = ff**2 * rr
    k_integrand = dcoef * (ffr**2 + angular) * rr
    m_integrand = ff**4 * rr
    nval = 2.0 * math.pi * torch.trapz(n_integrand, rr)
    kval = 2.0 * math.pi * torch.trapz(k_integrand, rr)
    mval = 2.0 * math.pi * torch.trapz(m_integrand, rr)
    scale = (beta * nval + 0.5 * kval) / (mval + torch.finfo(r.dtype).eps)
    return nval, kval, mval, scale


def xpinn_interface_loss(model: RadialVortexPINN, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if model.subdomains <= 1:
        return torch.zeros((), device=device, dtype=dtype)
    loss = torch.zeros((), device=device, dtype=dtype)
    for i in range(model.subdomains - 1):
        r_if = model.edges[i + 1].reshape(1, 1).detach().clone().to(device=device, dtype=dtype)
        r_if.requires_grad_(True)
        left = model.forward_subdomain(r_if, i)
        right = model.forward_subdomain(r_if, i + 1)
        left_r = gradients(left, r_if)
        right_r = gradients(right, r_if)
        loss = loss + torch.mean((left - right) ** 2) + torch.mean((left_r - right_r) ** 2)
    return loss / float(model.subdomains - 1)


def train_pinn(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> TrainResult:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    r_max = math.sqrt(2.0) * args.L
    eps_r = max(1.0e-5, r_max * 1.0e-5)
    lambda_tail = math.sqrt(2.0 * args.beta / args.D)

    model = RadialVortexPINN(
        r_max, args.beta, args.D, args.width, args.depth, args.xpinn_subdomains
    ).to(
        device=device, dtype=dtype
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history = []
    epoch_times = []
    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        sync_if_needed(device)
        epoch_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        r_int = eps_r + (r_max - 2.0 * eps_r) * torch.rand(
            args.collocation, 1, device=device, dtype=dtype
        )
        r_int.requires_grad_(True)
        f = model(r_int)
        fr = gradients(f, r_int)
        frr = gradients(fr, r_int)
        radial_laplacian = frr + fr / r_int - (args.q**2) * f / (r_int**2)
        residual = 0.5 * args.D * radial_laplacian + f**3 - args.beta * f
        loss_pde = torch.mean(residual**2)

        r_b = torch.full((1, 1), r_max, device=device, dtype=dtype, requires_grad=True)
        f_b = model(r_b)
        fr_b = gradients(f_b, r_b)
        loss_outer_robin = torch.mean((fr_b + lambda_tail * f_b) ** 2)
        loss_interface = xpinn_interface_loss(model, device, dtype)
        loss_robin = loss_outer_robin + args.xpinn_interface_weight * loss_interface

        r_quad = torch.linspace(eps_r, r_max, args.quad_points, device=device, dtype=dtype)
        r_quad = r_quad.reshape(-1, 1)
        r_quad.requires_grad_(True)
        nval, kval, mval, scale = radial_quantities(
            model, r_quad, args.beta, args.D, args.q
        )
        loss_scale = torch.zeros((), device=device, dtype=dtype)
        identity_error = torch.abs(mval - 0.5 * kval - args.beta * nval)

        total_loss = (
            args.loss_pde_weight * loss_pde
            + args.loss_robin_weight * loss_robin
        )
        total_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        sync_if_needed(device)
        elapsed = time.perf_counter() - epoch_start
        epoch_times.append(elapsed)
        row = [
            epoch,
            float(total_loss.detach().cpu()),
            float(loss_pde.detach().cpu()),
            float(loss_robin.detach().cpu()),
            float(loss_scale.detach().cpu()),
            float(scale.detach().cpu()),
            float(nval.detach().cpu()),
            float(kval.detach().cpu()),
            float(mval.detach().cpu()),
            float(identity_error.detach().cpu()),
            optimizer.param_groups[0]["lr"],
            elapsed,
        ]
        history.append(row)

        if args.print_every > 0 and (epoch == 1 or epoch % args.print_every == 0):
            print(
                "epoch={:6d} total={:.6e} pde={:.6e} robin={:.6e} "
                "scale={:.6e} S={:.6e}".format(
                    epoch, row[1], row[2], row[3], row[4], row[5]
                ),
                flush=True,
            )

    sync_if_needed(device)
    total_training_seconds = time.perf_counter() - train_start
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_allocated = 0
        peak_reserved = 0

    history_array = np.asarray(history, dtype=float)
    epoch_seconds = np.asarray(epoch_times, dtype=float)
    return TrainResult(
        model=model,
        history=history_array,
        total_training_seconds=total_training_seconds,
        epoch_seconds=epoch_seconds,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        final_total_loss=float(history_array[-1, 1]),
    )


def load_xyz_grid(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",")
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"{path} is not an x,y,value table.")
    x = np.unique(data[:, 0])
    y = np.unique(data[:, 1])
    values = np.empty((y.size, x.size), dtype=float)
    ix = np.searchsorted(x, data[:, 0])
    iy = np.searchsorted(y, data[:, 1])
    values[iy, ix] = data[:, 2]
    return x, y, values


def load_true_fields(fdm_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    x_ref = None
    y_ref = None
    fields: Dict[str, np.ndarray] = {}
    for name, filename, _, _ in FIELD_SPECS:
        x, y, value = load_xyz_grid(fdm_dir / filename)
        if x_ref is None:
            x_ref = x
            y_ref = y
        elif not (np.allclose(x_ref, x) and np.allclose(y_ref, y)):
            raise ValueError(f"{filename} is not on the same grid as the other fields.")
        fields[name] = value
    assert x_ref is not None and y_ref is not None
    return x_ref, y_ref, fields


def evaluate_phi(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    chunk_size: int = 131072,
) -> np.ndarray:
    model.eval()
    x_grid, y_grid = np.meshgrid(x, y)
    rr = np.sqrt(x_grid**2 + y_grid**2)
    theta = np.arctan2(y_grid, x_grid)
    flat_r = rr.reshape(-1)
    pieces = []
    with torch.no_grad():
        for start in range(0, flat_r.size, chunk_size):
            stop = min(start + chunk_size, flat_r.size)
            r_tensor = torch.as_tensor(
                flat_r[start:stop, None], device=device, dtype=dtype
            )
            f_piece = model(r_tensor).detach().cpu().numpy().reshape(-1)
            pieces.append(f_piece)
    amp = np.concatenate(pieces).reshape(rr.shape)
    phi = amp * np.exp(1j * args.q * theta)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    phi[rr < 0.5 * min(dx, dy)] = 0.0
    return phi


def make_predicted_fields(
    phi: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    beta: float,
    dcoef: float,
) -> Dict[str, np.ndarray]:
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    abs_phi = np.abs(phi)
    intensity = abs_phi**2
    phase = np.angle(phi)
    phi_y, phi_x = np.gradient(phi, dy, dx)
    grad_sq = np.abs(phi_x) ** 2 + np.abs(phi_y) ** 2
    n_density = intensity
    k_density = dcoef * grad_sq
    m_density = intensity**2
    e_nl = 0.5 * m_density
    h_density = 0.5 * k_density - 0.5 * m_density
    eta = np.abs(m_density - 0.5 * k_density - beta * n_density)
    return {
        "abs_phi": abs_phi,
        "abs_phi_squared": intensity,
        "arg_phi": phase,
        "e_nl": e_nl,
        "h_density": h_density,
        "n_density": n_density,
        "k_density": k_density,
        "m_density": m_density,
        "identity_density_eta": eta,
    }


def phase_difference(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (pred - true)))


def field_difference(pred: np.ndarray, true: np.ndarray, kind: str) -> np.ndarray:
    if kind == "phase":
        return phase_difference(pred, true)
    return pred - true


def compute_metrics(diff: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    diff_flat = diff.reshape(-1)
    true_flat = true.reshape(-1)
    mse = float(np.mean(diff_flat**2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(diff_flat)))
    l2 = float(np.linalg.norm(diff_flat) / (np.linalg.norm(true_flat) + np.finfo(float).eps))
    max_error = float(np.max(np.abs(diff_flat)))
    return {
        "RMSE": rmse,
        "MSE": mse,
        "MAE": mae,
        "L2RelativeError": l2,
        "MaxError": max_error,
    }


def integrate_2d(value: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(trapezoid(value, x, axis=1), y, axis=0))


def compute_global_quantities(
    fields: Dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    beta: float,
) -> Dict[str, float]:
    nval = integrate_2d(fields["n_density"], x, y)
    kval = integrate_2d(fields["k_density"], x, y)
    mval = integrate_2d(fields["m_density"], x, y)
    hval = 0.5 * kval - 0.5 * mval
    identity = abs(mval - 0.5 * kval - beta * nval)
    return {
        "N": nval,
        "K": kval,
        "M": mval,
        "H": hval,
        "abs_M_minus_K_over_2_minus_beta_N": identity,
    }


def compare_global_quantities(
    true_global: Dict[str, float],
    pred_global: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    compared = {}
    for name in true_global:
        true_value = true_global[name]
        pred_value = pred_global[name]
        abs_error = abs(pred_value - true_value)
        rel_error = abs_error / (abs(true_value) + np.finfo(float).eps)
        compared[name] = {
            "True": true_value,
            "PINN": pred_value,
            "AbsError": abs_error,
            "RelError": rel_error,
        }
    return compared


def save_global_quantities(
    path: Path,
    global_metrics: Dict[str, Dict[str, float]],
) -> None:
    rows = []
    for name, values in global_metrics.items():
        rows.append(
            [
                name,
                values["True"],
                values["PINN"],
                values["AbsError"],
                values["RelError"],
            ]
        )
    with path.open("w", encoding="utf-8-sig") as f:
        f.write("Quantity True PINN AbsError RelError\n")
        for row in rows:
            f.write(
                "{} {:.16e} {:.16e} {:.16e} {:.16e}\n".format(
                    row[0], row[1], row[2], row[3], row[4]
                )
            )


def interp2_uniform(
    x: np.ndarray,
    y: np.ndarray,
    value: np.ndarray,
    xq: np.ndarray,
    yq: np.ndarray,
) -> np.ndarray:
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    tx = (xq - x[0]) / dx
    ty = (yq - y[0]) / dy
    ix0 = np.floor(tx).astype(int)
    iy0 = np.floor(ty).astype(int)
    ix0 = np.clip(ix0, 0, len(x) - 2)
    iy0 = np.clip(iy0, 0, len(y) - 2)
    wx = tx - ix0
    wy = ty - iy0
    v00 = value[iy0, ix0]
    v10 = value[iy0, ix0 + 1]
    v01 = value[iy0 + 1, ix0]
    v11 = value[iy0 + 1, ix0 + 1]
    return (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v10
        + (1.0 - wx) * wy * v01
        + wx * wy * v11
    )


def save_line_xyz_txt(
    path: Path,
    x_line: np.ndarray,
    y_line: np.ndarray,
    value: np.ndarray,
) -> None:
    table = np.column_stack((x_line.reshape(-1), y_line.reshape(-1), value.reshape(-1)))
    np.savetxt(path, table, fmt="%.16e")


def save_line_plot(
    path: Path,
    coordinate: np.ndarray,
    value: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    dpi: int,
) -> None:
    color = plt.get_cmap("jet")(0.15)
    plt.figure(figsize=(6.8, 4.6))
    plt.plot(coordinate, value, color=color, linewidth=1.8)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def save_line_comparison_plot(
    path: Path,
    coordinate: np.ndarray,
    true_value: np.ndarray,
    pred_value: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    dpi: int,
) -> None:
    colors = plt.get_cmap("jet")(np.linspace(0.12, 0.86, 2))
    plt.figure(figsize=(6.8, 4.6))
    plt.plot(coordinate, true_value, color=colors[0], linewidth=1.8, label="FDM true")
    plt.plot(
        coordinate,
        pred_value,
        color=colors[1],
        linewidth=1.6,
        linestyle="--",
        label="PINN predicted",
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def save_profile_metric_table(
    path: Path,
    profile_metrics: Dict[str, Dict[str, float]],
) -> None:
    with path.open("w", encoding="utf-8-sig") as f:
        f.write("Profile RMSE MSE MAE L2RelativeError MaxError\n")
        for name, values in profile_metrics.items():
            f.write(
                "{} {:.16e} {:.16e} {:.16e} {:.16e} {:.16e}\n".format(
                    name,
                    values["RMSE"],
                    values["MSE"],
                    values["MAE"],
                    values["L2RelativeError"],
                    values["MaxError"],
                )
            )


def save_radial_profile_analysis(
    out_dir: Path,
    x: np.ndarray,
    y: np.ndarray,
    true_fields: Dict[str, np.ndarray],
    pred_fields: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    abs_true = true_fields["abs_phi"]
    abs_pred = pred_fields["abs_phi"]
    ix0 = int(np.argmin(np.abs(x)))
    iy0 = int(np.argmin(np.abs(y)))

    profiles = (
        (
            "abs_phi_x_0",
            x,
            x,
            np.zeros_like(x),
            abs_true[iy0, :],
            abs_pred[iy0, :],
            "x",
            "|phi(x,0)|",
        ),
        (
            "abs_phi_0_y",
            y,
            np.zeros_like(y),
            y,
            abs_true[:, ix0],
            abs_pred[:, ix0],
            "y",
            "|phi(0,y)|",
        ),
    )

    for name, coordinate, x_line, y_line, true_value, pred_value, x_label, y_label in profiles:
        diff = pred_value - true_value
        abs_error = np.abs(diff)
        metrics[f"profile_{name}"] = compute_metrics(diff, true_value)
        base = out_dir / f"Kerr2D_PINN_profile_{name}"
        save_line_xyz_txt(base.with_name(base.name + "_true.txt"), x_line, y_line, true_value)
        save_line_xyz_txt(base.with_name(base.name + "_pred.txt"), x_line, y_line, pred_value)
        save_line_xyz_txt(
            base.with_name(base.name + "_Maxerror.txt"), x_line, y_line, abs_error
        )
        comparison_table = np.column_stack((x_line, y_line, true_value, pred_value, abs_error))
        np.savetxt(
            base.with_name(base.name + "_comparison.txt"),
            comparison_table,
            header="x y true pred abs_error",
            fmt="%.16e",
        )
        save_line_plot(
            base.with_name(base.name + "_true.png"),
            coordinate,
            true_value,
            x_label,
            y_label,
            f"True {y_label}",
            args.plot_dpi,
        )
        save_line_plot(
            base.with_name(base.name + "_pred.png"),
            coordinate,
            pred_value,
            x_label,
            y_label,
            f"PINN predicted {y_label}",
            args.plot_dpi,
        )
        save_line_plot(
            base.with_name(base.name + "_Maxerror.png"),
            coordinate,
            abs_error,
            x_label,
            "absolute error",
            f"Maxerror {y_label}, max={metrics[f'profile_{name}']['MaxError']:.6e}",
            args.plot_dpi,
        )
        save_line_comparison_plot(
            base.with_name(base.name + "_comparison.png"),
            coordinate,
            true_value,
            pred_value,
            x_label,
            y_label,
            f"{y_label}: FDM true vs PINN predicted",
            args.plot_dpi,
        )

    r = np.linspace(0.0, args.L, DEFAULT_RADIAL_PROFILE_POINTS)
    x_line = r
    y_line = np.zeros_like(r)
    true_r = interp2_uniform(x, y, abs_true, x_line, y_line)
    pred_r = interp2_uniform(x, y, abs_pred, x_line, y_line)
    diff_r = pred_r - true_r
    abs_error_r = np.abs(diff_r)
    metrics["profile_abs_phi_r"] = compute_metrics(diff_r, true_r)
    base = out_dir / "Kerr2D_PINN_profile_abs_phi_r"
    save_line_xyz_txt(base.with_name(base.name + "_true.txt"), x_line, y_line, true_r)
    save_line_xyz_txt(base.with_name(base.name + "_pred.txt"), x_line, y_line, pred_r)
    save_line_xyz_txt(base.with_name(base.name + "_Maxerror.txt"), x_line, y_line, abs_error_r)
    np.savetxt(
        base.with_name(base.name + "_comparison.txt"),
        np.column_stack((x_line, y_line, true_r, pred_r, abs_error_r)),
        header="r y true pred abs_error",
        fmt="%.16e",
    )
    save_line_plot(
        base.with_name(base.name + "_true.png"),
        r,
        true_r,
        "r",
        "|phi(r)|",
        "True |phi(r)|",
        args.plot_dpi,
    )
    save_line_plot(
        base.with_name(base.name + "_pred.png"),
        r,
        pred_r,
        "r",
        "|phi(r)|",
        "PINN predicted |phi(r)|",
        args.plot_dpi,
    )
    save_line_plot(
        base.with_name(base.name + "_Maxerror.png"),
        r,
        abs_error_r,
        "r",
        "absolute error",
        f"Maxerror |phi(r)|, max={metrics['profile_abs_phi_r']['MaxError']:.6e}",
        args.plot_dpi,
    )
    save_line_comparison_plot(
        base.with_name(base.name + "_comparison.png"),
        r,
        true_r,
        pred_r,
        "r",
        "|phi(r)|",
        "|phi(r)|: FDM true vs PINN predicted",
        args.plot_dpi,
    )

    field_diff = abs_pred - abs_true
    field_abs_error = np.abs(field_diff)
    metrics["radial_abs_phi_field"] = compute_metrics(field_diff, abs_true)
    vmin = float(min(np.nanmin(abs_true), np.nanmin(abs_pred)))
    vmax = float(max(np.nanmax(abs_true), np.nanmax(abs_pred)))
    for tag, value, title in (
        ("true", abs_true, "True |phi(r)| field"),
        ("pred", abs_pred, "PINN predicted |phi(r)| field"),
    ):
        base_field = out_dir / f"Kerr2D_PINN_radial_abs_phi_field_{tag}"
        save_xyz_txt(base_field.with_suffix(".txt"), x, y, value)
        save_field_plot(
            base_field.with_suffix(".png"),
            x,
            y,
            value,
            title,
            args.plot_dpi,
            vmin=vmin,
            vmax=vmax,
        )
    base_error = out_dir / "Kerr2D_PINN_radial_abs_phi_field_Maxerror"
    save_xyz_txt(base_error.with_suffix(".txt"), x, y, field_abs_error)
    save_field_plot(
        base_error.with_suffix(".png"),
        x,
        y,
        field_abs_error,
        (
            "Maxerror |phi(r)| field, "
            f"max={metrics['radial_abs_phi_field']['MaxError']:.6e}"
        ),
        args.plot_dpi,
        vmin=0.0,
        vmax=float(np.nanmax(field_abs_error)),
    )

    save_profile_metric_table(out_dir / "Kerr2D_PINN_radial_profile_metrics.txt", metrics)
    return metrics


def phase_winding_number(
    x: np.ndarray,
    y: np.ndarray,
    phase: np.ndarray,
    radius: float,
    samples: int,
) -> float:
    theta = np.linspace(0.0, 2.0 * math.pi, samples + 1)
    xq = radius * np.cos(theta)
    yq = radius * np.sin(theta)
    phase_unit = np.exp(1j * phase)
    sampled_unit = interp2_uniform(x, y, phase_unit, xq, yq)
    sampled_phase = np.unwrap(np.angle(sampled_unit))
    return float((sampled_phase[-1] - sampled_phase[0]) / (2.0 * math.pi))


def save_topological_charge_validation(
    out_dir: Path,
    x: np.ndarray,
    y: np.ndarray,
    true_fields: Dict[str, np.ndarray],
    pred_fields: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Dict[str, float]:
    radii = np.asarray(DEFAULT_TOPOLOGY_RADIUS_FRACTIONS, dtype=float) * args.L
    rows = []
    for radius in radii:
        true_qgamma = phase_winding_number(
            x, y, true_fields["arg_phi"], float(radius), DEFAULT_TOPOLOGY_SAMPLES
        )
        pred_qgamma = phase_winding_number(
            x, y, pred_fields["arg_phi"], float(radius), DEFAULT_TOPOLOGY_SAMPLES
        )
        rows.append(
            [
                radius,
                true_qgamma,
                pred_qgamma,
                float(args.q),
                abs(true_qgamma - args.q),
                abs(pred_qgamma - args.q),
            ]
        )
    table = np.asarray(rows, dtype=float)
    header = "radius true_Q_gamma pred_Q_gamma target_q true_abs_error pred_abs_error"
    np.savetxt(out_dir / "Kerr2D_PINN_topological_charge.txt", table, header=header)

    colors = plt.get_cmap("jet")(np.linspace(0.12, 0.90, 3))
    plt.figure(figsize=(6.8, 4.6))
    plt.plot(table[:, 0], table[:, 1], "-o", color=colors[0], label="FDM true Q_gamma")
    plt.plot(
        table[:, 0],
        table[:, 2],
        "-s",
        color=colors[1],
        label="PINN predicted Q_gamma",
    )
    plt.axhline(args.q, color=colors[2], linestyle="--", label=f"target q={args.q}")
    plt.xlabel("radius")
    plt.ylabel("Q_gamma")
    plt.title("Topological charge validation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "Kerr2D_PINN_topological_charge.png", dpi=args.plot_dpi)
    plt.close()

    return {
        "TargetQ": float(args.q),
        "TrueMeanAbsError": float(np.mean(table[:, 4])),
        "TrueMaxAbsError": float(np.max(table[:, 4])),
        "PINNMeanAbsError": float(np.mean(table[:, 5])),
        "PINNMaxAbsError": float(np.max(table[:, 5])),
        "SampleCountPerCircle": float(DEFAULT_TOPOLOGY_SAMPLES),
        "RadiusCount": float(len(radii)),
    }


def save_structure_convergence(
    out_dir: Path,
    history: np.ndarray,
    true_global: Dict[str, float],
    beta: float,
    dpi: int,
) -> None:
    fdm_n = true_global["N"]
    fdm_k = true_global["K"]
    fdm_m = true_global["M"]
    fdm_s = (beta * fdm_n + 0.5 * fdm_k) / (fdm_m + np.finfo(float).eps)
    fdm_identity = true_global["abs_M_minus_K_over_2_minus_beta_N"]
    epoch = history[:, 0]
    convergence_table = np.column_stack(
        (
            epoch,
            history[:, 6],
            history[:, 7],
            history[:, 8],
            history[:, 5],
            history[:, 9],
            np.full_like(epoch, fdm_n),
            np.full_like(epoch, fdm_k),
            np.full_like(epoch, fdm_m),
            np.full_like(epoch, fdm_s),
            np.full_like(epoch, fdm_identity),
        )
    )
    header = (
        "Epoch N K M S IdentityError "
        "FDM_N FDM_K FDM_M FDM_S FDM_IdentityError"
    )
    np.savetxt(
        out_dir / "Kerr2D_PINN_structure_convergence.txt",
        convergence_table,
        header=header,
    )

    curve_specs = (
        ("N", 6, fdm_n, "Kerr2D_PINN_structure_convergence_N.png", "N"),
        ("K", 7, fdm_k, "Kerr2D_PINN_structure_convergence_K.png", "K"),
        ("M", 8, fdm_m, "Kerr2D_PINN_structure_convergence_M.png", "M"),
        ("S", 5, fdm_s, "Kerr2D_PINN_structure_convergence_S.png", "S"),
        (
            "|M-K/2-beta N|",
            9,
            fdm_identity,
            "Kerr2D_PINN_structure_convergence_identity.png",
            "Identity error",
        ),
    )
    colors = plt.get_cmap("jet")(np.linspace(0.12, 0.90, 2))
    for label, col, fdm_value, filename, ylabel in curve_specs:
        plt.figure(figsize=(7.0, 4.6))
        plt.plot(epoch, history[:, col], color=colors[0], linewidth=1.6, label="PINN")
        plt.axhline(
            fdm_value,
            color=colors[1],
            linestyle="--",
            linewidth=1.5,
            label=f"FDM {label}={fdm_value:.6e}",
        )
        plt.xlabel("epoch e")
        plt.ylabel(ylabel)
        plt.title(f"Structure convergence: {label}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=dpi)
        plt.close()

    identity = np.maximum(history[:, 9], np.finfo(float).tiny)
    plt.figure(figsize=(7.0, 4.6))
    plt.semilogy(epoch, identity, color=colors[0], linewidth=1.6, label="PINN")
    plt.axhline(
        max(fdm_identity, np.finfo(float).tiny),
        color=colors[1],
        linestyle="--",
        linewidth=1.5,
        label=f"FDM={fdm_identity:.6e}",
    )
    plt.xlabel("epoch e")
    plt.ylabel("|M-K/2-beta N|")
    plt.title("Structure identity convergence (log scale)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        out_dir / "Kerr2D_PINN_structure_convergence_identity_log.png",
        dpi=dpi,
    )
    plt.close()


def save_xyz_txt(path: Path, x: np.ndarray, y: np.ndarray, value: np.ndarray) -> None:
    x_grid, y_grid = np.meshgrid(x, y)
    table = np.column_stack((x_grid.reshape(-1), y_grid.reshape(-1), value.reshape(-1)))
    np.savetxt(path, table, fmt="%.16e")


def save_field_plot(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    value: np.ndarray,
    title: str,
    dpi: int,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    if vmin is not None and vmax is not None and np.isclose(vmin, vmax):
        pad = 1.0 if np.isclose(vmin, 0.0) else 0.01 * abs(vmin)
        vmin -= pad
        vmax += pad
    plt.figure(figsize=(6.0, 5.2))
    plt.imshow(
        value,
        origin="lower",
        extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
        aspect="equal",
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def save_loss_history(out_dir: Path, history: np.ndarray, dpi: int) -> None:
    header = (
        "Epoch TotalLoss PDELoss RobinBoundaryLoss PetviashviliScaleLoss "
        "PetviashviliS N K M IdentityError LearningRate EpochSeconds"
    )
    np.savetxt(out_dir / "Kerr2D_PINN_loss_history.txt", history, header=header)

    labels = (
        ("Total", 1),
        ("PDE", 2),
        ("Robin", 3),
        ("PetviashviliScale", 4),
    )
    colors = plt.get_cmap("jet")(np.linspace(0.05, 0.95, len(labels)))

    for log_y, suffix in ((False, "linear"), (True, "log")):
        plt.figure(figsize=(7.2, 4.8))
        for color, (label, col) in zip(colors, labels):
            values = np.maximum(history[:, col], np.finfo(float).tiny)
            plt.plot(history[:, 0], values, label=label, color=color, linewidth=1.4)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        if log_y:
            plt.yscale("log")
        plt.title(f"PINN loss history ({suffix})")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"Kerr2D_PINN_loss_{suffix}.png", dpi=dpi)
        plt.close()


def save_outputs(
    out_dir: Path,
    x: np.ndarray,
    y: np.ndarray,
    true_fields: Dict[str, np.ndarray],
    pred_fields: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Dict[str, Dict[str, float]]:
    metrics = {}
    for name, _, label, kind in FIELD_SPECS:
        true_value = true_fields[name]
        pred_value = pred_fields[name]
        diff = field_difference(pred_value, true_value, kind)
        abs_error = np.abs(diff)
        metrics[name] = compute_metrics(diff, true_value)

        if kind == "phase":
            vmin, vmax = -math.pi, math.pi
        else:
            vmin = float(min(np.nanmin(true_value), np.nanmin(pred_value)))
            vmax = float(max(np.nanmax(true_value), np.nanmax(pred_value)))

        for tag, value, title in (
            ("true", true_value, f"True {label}"),
            ("pred", pred_value, f"PINN predicted {label}"),
        ):
            base = out_dir / f"Kerr2D_PINN_{name}_{tag}"
            save_xyz_txt(base.with_suffix(".txt"), x, y, value)
            save_field_plot(
                base.with_suffix(".png"),
                x,
                y,
                value,
                title,
                args.plot_dpi,
                vmin=vmin,
                vmax=vmax,
            )

        error_base = out_dir / f"Kerr2D_PINN_{name}_Maxerror"
        save_xyz_txt(error_base.with_suffix(".txt"), x, y, abs_error)
        save_field_plot(
            error_base.with_suffix(".png"),
            x,
            y,
            abs_error,
            f"Maxerror {label}, max={metrics[name]['MaxError']:.6e}",
            args.plot_dpi,
            vmin=0.0,
            vmax=float(np.nanmax(abs_error)),
        )
    return metrics


def write_metrics_file(
    path: Path,
    args: argparse.Namespace,
    fdm_dir: Path,
    train: TrainResult,
    eval_seconds: float,
    eval_grid: Tuple[int, int],
    field_metrics: Dict[str, Dict[str, float]],
    global_metrics: Dict[str, Dict[str, float]],
    profile_metrics: Dict[str, Dict[str, float]],
    topology_metrics: Dict[str, float],
    device: torch.device,
) -> None:
    epoch_seconds = train.epoch_seconds
    summary = {
        "Device": str(device),
        "Seed": args.seed,
        "Beta": args.beta,
        "D": args.D,
        "q": args.q,
        "L": args.L,
        "Rrad": math.sqrt(2.0) * args.L,
        "Epochs": args.epochs,
        "CollocationPointsPerEpoch": args.collocation,
        "QuadraturePoints": args.quad_points,
        "FDMDirectory": str(fdm_dir),
        "EvaluationGrid": f"{eval_grid[0]}x{eval_grid[1]}",
        "EvaluationPointCount": int(eval_grid[0] * eval_grid[1]),
        "TotalTrainingSeconds": train.total_training_seconds,
        "EvaluationSeconds": eval_seconds,
        "MeanEpochSeconds": float(np.mean(epoch_seconds)),
        "StdEpochSeconds": float(np.std(epoch_seconds)),
        "MedianEpochSeconds": float(np.median(epoch_seconds)),
        "MinEpochSeconds": float(np.min(epoch_seconds)),
        "MaxEpochSeconds": float(np.max(epoch_seconds)),
        "PeakTrainMemoryAllocatedBytes": train.peak_allocated_bytes,
        "PeakTrainMemoryAllocatedMiB": train.peak_allocated_bytes / (1024.0**2),
        "PeakTrainMemoryReservedBytes": train.peak_reserved_bytes,
        "PeakTrainMemoryReservedMiB": train.peak_reserved_bytes / (1024.0**2),
        "FinalTotalLoss": train.final_total_loss,
    }

    with path.open("w", encoding="utf-8-sig") as f:
        f.write("# Kerr2D XPINN run metrics\n")
        for key, value in summary.items():
            f.write(f"{key} {value}\n")
        f.write("\n# Field RMSE MSE MAE L2RelativeError MaxError\n")
        for field_name, values in field_metrics.items():
            f.write(
                "{} {:.16e} {:.16e} {:.16e} {:.16e} {:.16e}\n".format(
                    field_name,
                    values["RMSE"],
                    values["MSE"],
                    values["MAE"],
                    values["L2RelativeError"],
                    values["MaxError"],
                )
            )
        f.write("\n# GlobalQuantity True PINN AbsError RelError\n")
        for quantity_name, values in global_metrics.items():
            f.write(
                "{} {:.16e} {:.16e} {:.16e} {:.16e}\n".format(
                    quantity_name,
                    values["True"],
                    values["PINN"],
                    values["AbsError"],
                    values["RelError"],
                )
            )
        f.write("\n# RadialProfile RMSE MSE MAE L2RelativeError MaxError\n")
        for profile_name, values in profile_metrics.items():
            f.write(
                "{} {:.16e} {:.16e} {:.16e} {:.16e} {:.16e}\n".format(
                    profile_name,
                    values["RMSE"],
                    values["MSE"],
                    values["MAE"],
                    values["L2RelativeError"],
                    values["MaxError"],
                )
            )
        f.write("\n# TopologicalChargeValidation\n")
        for key, value in topology_metrics.items():
            f.write(f"{key} {value}\n")


def save_config(out_dir: Path, args: argparse.Namespace, fdm_dir: Path, device: torch.device) -> None:
    config = vars(args).copy()
    config["fdm_dir"] = str(fdm_dir)
    config["out_dir"] = str(out_dir)
    config["device_resolved"] = str(device)
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    with (out_dir / "Kerr2D_PINN_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def main() -> None:
    args = parse_args()
    script_path = Path(__file__)
    fdm_dir = args.fdm_dir.resolve() if args.fdm_dir else infer_fdm_dir(script_path)
    out_dir = args.out_dir.resolve() if args.out_dir else script_path.resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = choose_device(args.device)

    print(f"FDM truth directory: {fdm_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Device: {device}, dtype: {args.dtype}")
    save_config(out_dir, args, fdm_dir, device)

    train = train_pinn(args, device, dtype)
    save_loss_history(out_dir, train.history, args.plot_dpi)
    torch.save(
        {
            "model_state_dict": train.model.state_dict(),
            "args": vars(args),
            "history": train.history,
        },
        out_dir / "Kerr2D_PINN_model.pt",
    )

    eval_start = time.perf_counter()
    x, y, true_fields = load_true_fields(fdm_dir)
    phi_pred = evaluate_phi(train.model, x, y, args, device, dtype)
    pred_fields = make_predicted_fields(phi_pred, x, y, args.beta, args.D)
    field_metrics = save_outputs(out_dir, x, y, true_fields, pred_fields, args)
    true_global = compute_global_quantities(true_fields, x, y, args.beta)
    pred_global = compute_global_quantities(pred_fields, x, y, args.beta)
    global_metrics = compare_global_quantities(true_global, pred_global)
    save_global_quantities(out_dir / "Kerr2D_PINN_global_quantities.txt", global_metrics)
    profile_metrics = save_radial_profile_analysis(
        out_dir, x, y, true_fields, pred_fields, args
    )
    topology_metrics = save_topological_charge_validation(
        out_dir, x, y, true_fields, pred_fields, args
    )
    save_structure_convergence(
        out_dir, train.history, true_global, args.beta, args.plot_dpi
    )
    eval_seconds = time.perf_counter() - eval_start

    write_metrics_file(
        out_dir / "Kerr2D_PINN_metrics.txt",
        args,
        fdm_dir,
        train,
        eval_seconds,
        (len(x), len(y)),
        field_metrics,
        global_metrics,
        profile_metrics,
        topology_metrics,
        device,
    )
    print("Finished.")
    print(f"Metrics: {out_dir / 'Kerr2D_PINN_metrics.txt'}")


if __name__ == "__main__":
    main()
