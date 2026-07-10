from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FDM_DIR = next(path for path in ROOT.iterdir() if path.is_dir() and path.name.startswith("FDM"))
OUTPUT = ROOT / "Kerr2D_FDM_all_results_redraw.png"
JET = plt.get_cmap("jet")


FIELD_PANELS = [
    ("Kerr2D_abs_phi", r"$|\phi(x,y)|$"),
    ("Kerr2D_abs_phi_squared", r"$|\phi(x,y)|^2$"),
    ("Kerr2D_arg_phi", r"$\arg\phi(x,y)$"),
    ("Kerr2D_e_nl", r"$e_{nl}(x,y)$"),
    ("Kerr2D_h_density", r"$h(x,y)$"),
    ("Kerr2D_n_density", r"$n(x,y)=|\phi|^2$"),
    ("Kerr2D_k_density", r"$k(x,y)=D|\nabla\phi|^2$"),
    ("Kerr2D_m_density", r"$m(x,y)=|\phi|^4$"),
    ("Kerr2D_identity_density_eta", r"$|m-k/2-\beta n|$"),
    ("Kerr2D_pde_residual_abs", r"$|R(x,y)|$"),
]

PANEL_NAMES = [
    *(name for name, _ in FIELD_PANELS),
    "Kerr2D_abs_phi_x_0",
    "Kerr2D_abs_phi_0_y",
    "Kerr2D_global_quantities",
    "Kerr2D_grid_independence_NKM",
    "Kerr2D_grid_independence_identity",
    "Kerr2D_observed_grid_error",
    "Kerr2D_observed_order",
    "Kerr2D_FDM_Laplacian_order",
]


def panel_label(index: int) -> str:
    return f"({chr(ord('a') + index)})"


def load_xyz_grid(stem: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(FDM_DIR / f"{stem}.csv", delimiter=",")
    x = np.unique(data[:, 0])
    y = np.unique(data[:, 1])
    value = data[:, 2].reshape((len(y), len(x)), order="F")
    return x, y, value


def load_xyz_line(stem: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(FDM_DIR / f"{stem}.csv", delimiter=",")
    coord = data[:, 0] if np.ptp(data[:, 0]) > np.ptp(data[:, 1]) else data[:, 1]
    return coord, data[:, 2]


def load_table(name: str) -> np.ndarray:
    return np.genfromtxt(FDM_DIR / name, delimiter=",")


def style_axis(ax: plt.Axes, title: str, label: str) -> None:
    ax.set_title(f"{label} {title}", loc="left", fontsize=10.5, pad=4)
    ax.tick_params(labelsize=8, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


def draw_field(ax: plt.Axes, stem: str, title: str, index: int) -> None:
    x, y, value = load_xyz_grid(stem)
    image = ax.imshow(
        value,
        extent=(x.min(), x.max(), y.min(), y.max()),
        origin="lower",
        aspect="equal",
        cmap="jet",
        interpolation="nearest",
    )
    style_axis(ax, title, panel_label(index))
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("y", fontsize=9)
    cb = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
    cb.ax.tick_params(labelsize=7, length=2)


def draw_abs_phi_x(ax: plt.Axes, index: int) -> None:
    x, value = load_xyz_line("Kerr2D_abs_phi_x_0")
    ax.plot(x, value, color=JET(0.18), linewidth=1.9)
    style_axis(ax, r"$|\phi(x,0)|$", panel_label(index))
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel(r"$|\phi|$", fontsize=9)
    ax.grid(True, alpha=0.28, linewidth=0.7)


def draw_abs_phi_y(ax: plt.Axes, index: int) -> None:
    y, value = load_xyz_line("Kerr2D_abs_phi_0_y")
    ax.plot(y, value, color=JET(0.82), linewidth=1.9)
    style_axis(ax, r"$|\phi(0,y)|$", panel_label(index))
    ax.set_xlabel("y", fontsize=9)
    ax.set_ylabel(r"$|\phi|$", fontsize=9)
    ax.grid(True, alpha=0.28, linewidth=0.7)


def draw_global_quantities(ax: plt.Axes, index: int) -> None:
    values = load_table("Kerr2D_global_quantities.csv")
    labels = ["N", "K", "M", "H", r"$|M-K/2-\beta N|$"]
    plotted = [values[3], values[4], values[5], values[6], values[7]]
    colors = [JET(v) for v in np.linspace(0.08, 0.92, len(plotted))]
    ax.bar(np.arange(len(plotted)), plotted, color=colors, width=0.72)
    ax.set_xticks(np.arange(len(plotted)), labels, rotation=25, ha="right", fontsize=8)
    style_axis(ax, "Global integral quantities", panel_label(index))
    ax.set_ylabel("value", fontsize=9)
    ax.grid(True, axis="y", alpha=0.28, linewidth=0.7)


def draw_grid_independence_nkm(ax: plt.Axes, index: int) -> None:
    data = load_table("Kerr2D_grid_independence_table.csv")
    dx = data[:, 1]
    ax.plot(dx, data[:, 4], "-o", color=JET(0.12), linewidth=1.7, markersize=4.2, label="N")
    ax.plot(dx, data[:, 5], "-s", color=JET(0.50), linewidth=1.7, markersize=4.2, label="K")
    ax.plot(dx, data[:, 6], "-^", color=JET(0.88), linewidth=1.7, markersize=4.2, label="M")
    ax.invert_xaxis()
    style_axis(ax, "Grid independence: N, K, M", panel_label(index))
    ax.set_xlabel("grid spacing dx", fontsize=9)
    ax.set_ylabel("integral value", fontsize=9)
    ax.grid(True, alpha=0.28, linewidth=0.7)
    ax.legend(fontsize=7.5, frameon=False, loc="best")


def draw_grid_identity(ax: plt.Axes, index: int) -> None:
    data = load_table("Kerr2D_grid_independence_table.csv")
    ax.semilogy(data[:, 1], data[:, 8], "-o", color=JET(0.2), linewidth=1.7, markersize=4.2)
    ax.invert_xaxis()
    style_axis(ax, r"Grid independence: $|M-K/2-\beta N|$", panel_label(index))
    ax.set_xlabel("grid spacing dx", fontsize=9)
    ax.set_ylabel(r"$|M-K/2-\beta N|$", fontsize=9)
    ax.grid(True, which="both", alpha=0.28, linewidth=0.7)


def draw_observed_grid_error(ax: plt.Axes, index: int) -> None:
    data = load_table("Kerr2D_observed_order_table.csv")
    valid = np.isfinite(data[:, 2])
    ax.loglog(data[valid, 1], data[valid, 2], "-o", color=JET(0.2), linewidth=1.7, markersize=4.2)
    style_axis(ax, r"Observed grid-refinement error", panel_label(index))
    ax.set_xlabel("grid spacing dx", fontsize=9)
    ax.set_ylabel(r"RMSE of $|\phi|$", fontsize=9)
    ax.grid(True, which="both", alpha=0.28, linewidth=0.7)


def draw_observed_order(ax: plt.Axes, index: int) -> None:
    data = load_table("Kerr2D_observed_order_table.csv")
    valid = np.isfinite(data[:, 3])
    ax.plot(
        data[valid, 1],
        data[valid, 3],
        "o",
        color=JET(0.85),
        markeredgecolor="black",
        markeredgewidth=0.45,
        markersize=8.0,
        label="observed order",
    )
    for dx, order in data[valid][:, [1, 3]]:
        ax.annotate(
            f"{order:.3f}",
            xy=(dx, order),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=JET(0.85),
        )
    ax.axhline(2, color=JET(0.55), linestyle="--", linewidth=1.1, label="Second order reference")
    style_axis(ax, "Observed order test", panel_label(index))
    ax.set_xlabel("coarse-grid dx", fontsize=9)
    ax.set_ylabel("observed order", fontsize=9)
    if np.any(valid):
        x0 = float(data[valid, 1][0])
        y_valid = data[valid, 3]
        ax.set_xlim(x0 - 0.018, x0 + 0.018)
        ax.set_ylim(1.88, max(2.85, float(np.nanmax(y_valid)) + 0.16))
    ax.grid(True, alpha=0.28, linewidth=0.7)
    ax.legend(fontsize=7.5, frameon=False, loc="best")


def draw_laplacian_order(ax: plt.Axes, index: int) -> None:
    data = load_table("Kerr2D_FDM_Laplacian_order_table.csv")
    ref = data[0, 2] * (data[:, 1] / data[0, 1]) ** 2
    ax.loglog(data[:, 1], data[:, 2], "-o", color=JET(0.2), linewidth=1.7, markersize=4.2, label="FDM Laplacian error")
    ax.loglog(data[:, 1], ref, "--", color=JET(0.85), linewidth=1.2, label=r"$O(dx^2)$ reference")
    style_axis(ax, "Manufactured-order test for FDM Laplacian", panel_label(index))
    ax.set_xlabel("grid spacing dx", fontsize=9)
    ax.set_ylabel("RMSE", fontsize=9)
    ax.grid(True, which="both", alpha=0.28, linewidth=0.7)
    ax.legend(fontsize=7.5, frameon=False, loc="best")


def verify_all_source_figures_are_redrawn() -> None:
    source_png_stems = {path.stem for path in FDM_DIR.glob("*.png")}
    panel_stems = set(PANEL_NAMES)
    missing = sorted(source_png_stems - panel_stems)
    extra = sorted(panel_stems - source_png_stems)
    if missing or extra:
        raise RuntimeError(f"Panel/source mismatch. Missing={missing}; extra={extra}")


def main() -> None:
    verify_all_source_figures_are_redrawn()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": 0.7,
            "savefig.bbox": "tight",
        }
    )

    fig, axes = plt.subplots(3, 6, figsize=(30, 15.2), dpi=220, constrained_layout=True)
    flat_axes = axes.ravel()

    index = 0
    for stem, title in FIELD_PANELS:
        draw_field(flat_axes[index], stem, title, index)
        index += 1

    draw_abs_phi_x(flat_axes[index], index)
    index += 1
    draw_abs_phi_y(flat_axes[index], index)
    index += 1
    draw_global_quantities(flat_axes[index], index)
    index += 1
    draw_grid_independence_nkm(flat_axes[index], index)
    index += 1
    draw_grid_identity(flat_axes[index], index)
    index += 1
    draw_observed_grid_error(flat_axes[index], index)
    index += 1
    draw_observed_order(flat_axes[index], index)
    index += 1
    draw_laplacian_order(flat_axes[index], index)

    fig.suptitle("Kerr2D FDM Results Redrawn from Source Data", fontsize=18, fontweight="bold")
    fig.savefig(OUTPUT, dpi=220)
    plt.close(fig)
    print(f"Saved {OUTPUT}")
    print(f"Redrawn panels: {len(PANEL_NAMES)}")


if __name__ == "__main__":
    main()
