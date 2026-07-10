from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


METHOD_ORDER = ("LNN-PINN", "PINN", "PI-VAST", "XPINN")
METHOD_COLORS = {
    "LNN-PINN": "#1f77b4",
    "PINN": "#d62728",
    "PI-VAST": "#ff7f0e",
    "XPINN": "#2ca02c",
}
METHOD_LINESTYLES = {
    "LNN-PINN": "-",
    "PINN": "--",
    "PI-VAST": "-.",
    "XPINN": ":",
}
METHOD_MARKERS = {
    "LNN-PINN": "o",
    "PINN": "s",
    "PI-VAST": "^",
    "XPINN": "D",
}
FDM_COLOR = "#000000"

FIELD_ORDER = (
    "abs_phi",
    "abs_phi_squared",
    "arg_phi",
    "e_nl",
    "h_density",
    "n_density",
    "k_density",
    "m_density",
    "identity_density_eta",
    "radial_abs_phi_field",
)

FIELD_LABELS = {
    "abs_phi": r"$|\phi|$",
    "abs_phi_squared": r"$|\phi|^2$",
    "arg_phi": r"$\arg(\phi)$",
    "e_nl": r"$e_{\mathrm{nl}}$",
    "h_density": r"$H$",
    "n_density": r"$N$",
    "k_density": r"$K$",
    "m_density": r"$M$",
    "identity_density_eta": r"$\eta$",
    "radial_abs_phi_field": r"radial $|\phi|$",
    "profile_abs_phi_x_0": r"$|\phi|(x,0)$",
    "profile_abs_phi_0_y": r"$|\phi|(0,y)$",
    "profile_abs_phi_r": r"$|\phi|(r)$",
}


@dataclass(frozen=True)
class MethodSpec:
    label: str
    out_dir: Path
    prefix: str
    color: str


def panel_label(index: int) -> str:
    index += 1
    chars: list[str] = []
    while index:
        index -= 1
        chars.append(chr(ord("a") + index % 26))
        index //= 26
    return "(" + "".join(reversed(chars)) + ")"


def add_panel_marker(
    ax: plt.Axes, index: int, fontsize: float = 6.0, outside: bool = False
) -> None:
    x, y = (-0.02, 1.045) if outside else (0.015, 0.985)
    va = "bottom" if outside else "top"
    ax.text(
        x,
        y,
        panel_label(index),
        transform=ax.transAxes,
        ha="left",
        va=va,
        fontsize=fontsize,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8},
        zorder=10,
    )


def detect_prefix(out_dir: Path) -> str:
    configs = sorted(out_dir.glob("*_config.json"))
    if configs:
        stem = configs[0].stem
        if stem.endswith("_config"):
            return stem[: -len("_config")]
    histories = sorted(out_dir.glob("*_loss_history.txt"))
    if histories:
        stem = histories[0].stem
        if stem.endswith("_loss_history"):
            return stem[: -len("_loss_history")]
    raise FileNotFoundError(f"Cannot detect output prefix in {out_dir}")


def discover_methods(root: Path, method_names: tuple[str, ...]) -> list[MethodSpec]:
    methods: list[MethodSpec] = []
    for name in method_names:
        out_dir = root / name / "outputs"
        if not out_dir.is_dir():
            raise FileNotFoundError(f"Missing output directory: {out_dir}")
        methods.append(
            MethodSpec(
                label=name,
                out_dir=out_dir,
                prefix=detect_prefix(out_dir),
                color=METHOD_COLORS.get(name, None) or "#333333",
            )
        )
    return methods


def load_txt(path: Path) -> np.ndarray:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def load_csv(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def field_file(method: MethodSpec, field: str, kind: str) -> Path:
    return method.out_dir / f"{method.prefix}_{field}_{kind}.txt"


def history_file(method: MethodSpec) -> Path:
    return method.out_dir / f"{method.prefix}_loss_history.txt"


def available_fields(methods: list[MethodSpec]) -> tuple[list[str], list[str]]:
    first = methods[0]
    fields: list[str] = []
    for true_file in sorted(first.out_dir.glob(f"{first.prefix}_*_true.txt")):
        field = true_file.name[len(first.prefix) + 1 : -len("_true.txt")]
        ok = True
        for method in methods:
            ok = ok and field_file(method, field, "pred").exists()
            ok = ok and field_file(method, field, "Maxerror").exists()
            ok = ok and field_file(method, field, "true").exists()
        if ok:
            fields.append(field)

    heat_fields: list[str] = []
    profile_fields: list[str] = []
    for field in fields:
        data = load_txt(field_file(first, field, "true"))
        if data.shape[1] < 3:
            continue
        xs = np.unique(data[:, 0])
        ys = np.unique(data[:, 1])
        if len(xs) > 1 and len(ys) > 1 and len(xs) * len(ys) == data.shape[0]:
            heat_fields.append(field)
        elif len(xs) > 1 or len(ys) > 1:
            profile_fields.append(field)

    heat_fields = [f for f in FIELD_ORDER if f in heat_fields] + sorted(
        f for f in heat_fields if f not in FIELD_ORDER
    )
    profile_fields = sorted(
        profile_fields,
        key=lambda f: (
            ["profile_abs_phi_x_0", "profile_abs_phi_0_y", "profile_abs_phi_r"].index(f)
            if f in ["profile_abs_phi_x_0", "profile_abs_phi_0_y", "profile_abs_phi_r"]
            else 99,
            f,
        ),
    )
    return heat_fields, profile_fields


def xyz_to_grid(data: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    xs = np.unique(x)
    ys = np.unique(y)
    grid = np.full((len(ys), len(xs)), np.nan, dtype=float)
    x_idx = np.searchsorted(xs, x)
    y_idx = np.searchsorted(ys, y)
    grid[y_idx, x_idx] = z
    extent = (float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()))
    return grid, extent


def line_from_xyz(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    xs = np.unique(data[:, 0])
    ys = np.unique(data[:, 1])
    if len(xs) >= len(ys):
        order = np.argsort(data[:, 0])
        return data[order, 0], data[order, 2], "x"
    order = np.argsort(data[:, 1])
    return data[order, 1], data[order, 2], "y"


def fdm_profile(root: Path, field: str) -> tuple[np.ndarray, np.ndarray, str]:
    fdm_dir = root / "FDM仿真"
    if field == "profile_abs_phi_x_0":
        data = load_csv(fdm_dir / "Kerr2D_abs_phi_x_0.csv")
        x = data[:, 0]
        axis_name = "x"
    elif field == "profile_abs_phi_0_y":
        data = load_csv(fdm_dir / "Kerr2D_abs_phi_0_y.csv")
        x = data[:, 1]
        axis_name = "y"
    elif field == "profile_abs_phi_r":
        data = load_csv(fdm_dir / "Kerr2D_abs_phi_x_0.csv")
        data = data[data[:, 0] >= 0.0]
        x = data[:, 0]
        axis_name = "r"
    else:
        raise FileNotFoundError(f"No FDM profile mapping for {field}")

    y = data[:, 2]
    order = np.argsort(x)
    return x[order], y[order], axis_name


def finite_range(arrays: list[np.ndarray], lower_percentile: float = 0.0) -> tuple[float, float]:
    vals = np.concatenate([np.ravel(a[np.isfinite(a)]) for a in arrays if np.isfinite(a).any()])
    if vals.size == 0:
        return 0.0, 1.0
    if lower_percentile:
        lo = float(np.nanpercentile(vals, lower_percentile))
    else:
        lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if math.isclose(lo, hi):
        pad = max(abs(hi), 1.0) * 0.05
        lo -= pad
        hi += pad
    return lo, hi


def error_norm(errors: list[np.ndarray]) -> colors.Normalize:
    vals = np.concatenate([np.ravel(a[np.isfinite(a)]) for a in errors if np.isfinite(a).any()])
    vals = vals[vals > 0.0]
    if vals.size == 0:
        return colors.Normalize(vmin=0.0, vmax=1.0)
    vmax = float(vals.max())
    vmin = max(float(vals.min()), vmax * 1e-8)
    if vmax / vmin > 100.0:
        return colors.LogNorm(vmin=vmin, vmax=vmax)
    return colors.Normalize(vmin=0.0, vmax=vmax)


def masked_for_norm(array: np.ndarray, norm: colors.Normalize) -> np.ndarray:
    if isinstance(norm, colors.LogNorm):
        return np.ma.masked_less_equal(array, 0.0)
    return array


def load_histories(methods: list[MethodSpec]) -> dict[str, np.ndarray]:
    histories: dict[str, np.ndarray] = {}
    for method in methods:
        data = load_txt(history_file(method))
        if data.shape[1] < 2:
            continue
        histories[method.label] = data[:, :2]
    return histories


def plot_loss_axis(ax: plt.Axes, methods: list[MethodSpec], histories: dict[str, np.ndarray]) -> None:
    for method in methods:
        hist = histories.get(method.label)
        if hist is None:
            continue
        y = np.where(hist[:, 1] > 0.0, hist[:, 1], np.nan)
        ax.plot(hist[:, 0], y, lw=0.75, color=method.color, label=method.label, alpha=0.95)
    ax.set_yscale("log")
    ax.set_ylabel("Loss", fontsize=8)
    ax.grid(True, which="both", lw=0.25, alpha=0.35)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=len(methods),
        fontsize=6,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.95,
        handlelength=1.8,
        columnspacing=1.2,
    )
    ax.tick_params(labelsize=7, length=2)
    add_panel_marker(ax, 0, fontsize=7)

    max_epoch = max(float(h[-1, 0]) for h in histories.values() if len(h))
    xmin = max_epoch * 0.78
    axins = inset_axes(ax, width="28%", height="48%", loc="upper right", borderpad=1.0)
    for method in methods:
        hist = histories.get(method.label)
        if hist is None:
            continue
        mask = hist[:, 0] >= xmin
        if mask.any():
            y = np.where(hist[mask, 1] > 0.0, hist[mask, 1], np.nan)
            axins.plot(hist[mask, 0], y, lw=0.6, color=method.color, alpha=0.95)
    axins.set_yscale("log")
    axins.set_title("zoom", fontsize=5, pad=1)
    axins.grid(True, which="both", lw=0.2, alpha=0.3)
    axins.tick_params(labelsize=5, length=1.5)


def apply_map_style(ax: plt.Axes, show_y_label: bool, field_label: str | None) -> None:
    ax.tick_params(labelsize=5, length=1.5, pad=1)
    ax.set_aspect("equal", adjustable="box")
    if field_label:
        ax.set_ylabel(field_label, fontsize=8)
    elif not show_y_label:
        ax.set_yticklabels([])


def save_figure(fig: plt.Figure, output_stem: Path, formats: tuple[str, ...], dpi: int) -> None:
    for fmt in formats:
        fig.savefig(output_stem.with_suffix("." + fmt), dpi=dpi, bbox_inches="tight")


def draw_profile_pair(
    root: Path,
    ax_val: plt.Axes,
    ax_err: plt.Axes,
    methods: list[MethodSpec],
    field: str,
    panel_idx: int,
    title_size: float = 8.0,
    tick_size: float = 7.0,
    show_legend: bool = True,
) -> int:
    label = FIELD_LABELS.get(field, field.replace("_", " "))
    try:
        x_ref, y_ref, axis_name = fdm_profile(root, field)
        ref_label = "FDM"
    except FileNotFoundError:
        x_ref, y_ref, axis_name = line_from_xyz(load_txt(field_file(methods[0], field, "true")))
        ref_label = "Reference"

    ax_val.plot(
        x_ref,
        y_ref,
        color=FDM_COLOR,
        ls="--",
        lw=1.25,
        alpha=0.68,
        label=ref_label,
        zorder=1,
    )
    add_panel_marker(ax_val, panel_idx, fontsize=max(5.2, tick_size), outside=True)
    ax_val.set_title(f"{label} comparison", fontsize=title_size, pad=6)
    panel_idx += 1

    for method_index, method in enumerate(methods):
        x_pred, y_pred, _ = line_from_xyz(load_txt(field_file(method, field, "pred")))
        markevery = max(1, len(x_pred) // 20)
        ax_val.plot(
            x_pred,
            y_pred,
            color=method.color,
            ls=METHOD_LINESTYLES.get(method.label, "-"),
            marker=METHOD_MARKERS.get(method.label, None),
            markevery=markevery,
            ms=2.4,
            markeredgewidth=0.35,
            lw=1.18,
            label=method.label,
            alpha=0.9,
            zorder=3 + method_index,
        )

        in_range = (x_pred >= x_ref.min()) & (x_pred <= x_ref.max())
        x_err = x_pred[in_range]
        ref_interp = np.interp(x_err, x_ref, y_ref)
        y_err = np.abs(y_pred[in_range] - ref_interp)
        y_err = np.where(y_err > 0.0, y_err, np.nan)
        ax_err.plot(
            x_err,
            y_err,
            color=method.color,
            ls=METHOD_LINESTYLES.get(method.label, "-"),
            marker=METHOD_MARKERS.get(method.label, None),
            markevery=max(1, len(x_err) // 18),
            ms=2.4,
            markeredgewidth=0.35,
            lw=1.18,
            label=method.label,
            alpha=0.9,
            zorder=3 + method_index,
        )

    add_panel_marker(ax_err, panel_idx, fontsize=max(5.2, tick_size), outside=True)
    ax_err.set_title(f"{label} max/absolute error", fontsize=title_size, pad=6)
    panel_idx += 1

    for ax in (ax_val, ax_err):
        ax.grid(True, lw=0.3, alpha=0.35)
        ax.tick_params(labelsize=tick_size, length=2)
        ax.set_xlabel(axis_name, fontsize=tick_size + 1)
    ax_val.set_ylabel(label, fontsize=tick_size + 1)
    ax_err.set_ylabel("Abs. error", fontsize=tick_size + 1)
    ax_err.set_yscale("log")
    if show_legend:
        ax_val.legend(
            fontsize=max(5.0, tick_size - 1),
            frameon=False,
            ncol=5,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.13),
            handlelength=2.2,
        )
        ax_err.legend(
            fontsize=max(5.0, tick_size - 1),
            frameon=False,
            ncol=4,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.13),
            handlelength=2.2,
        )
    return panel_idx


def draw_heatmap_figure(
    root: Path,
    methods: list[MethodSpec],
    histories: dict[str, np.ndarray],
    fields: list[str],
    output_stem: str,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    n_methods = len(methods)
    n_value = 1 + n_methods
    n_error = n_methods
    ncols = n_value + 1 + n_error + 1
    width_ratios = [1.0] * n_value + [0.026] + [1.0] * n_error + [0.026]
    height = max(5.4, 1.05 + 3.05 * len(fields))

    fig = plt.figure(figsize=(26.5, height))
    gs = GridSpec(
        1 + len(fields),
        ncols,
        figure=fig,
        height_ratios=[0.66] + [1.55] * len(fields),
        width_ratios=width_ratios,
        hspace=0.16,
        wspace=0.006,
    )

    loss_ax = fig.add_subplot(gs[0, :])
    plot_loss_axis(loss_ax, methods, histories)

    panel_idx = 1
    for row, field in enumerate(fields, start=1):
        true_grid, extent = xyz_to_grid(load_txt(field_file(methods[0], field, "true")))
        pred_grids = [xyz_to_grid(load_txt(field_file(m, field, "pred")))[0] for m in methods]
        err_grids = [xyz_to_grid(load_txt(field_file(m, field, "Maxerror")))[0] for m in methods]

        value_vmin, value_vmax = finite_range([true_grid] + pred_grids)
        value_norm = colors.Normalize(vmin=value_vmin, vmax=value_vmax)
        err_norm = error_norm(err_grids)
        label = FIELD_LABELS.get(field, field.replace("_", " "))

        value_axes = [fig.add_subplot(gs[row, col]) for col in range(n_value)]
        value_titles = ["Exact"] + [m.label for m in methods]
        value_arrays = [true_grid] + pred_grids
        last_im = None
        for ax, title, array in zip(value_axes, value_titles, value_arrays):
            last_im = ax.imshow(
                array,
                origin="lower",
                extent=extent,
                cmap="jet",
                norm=value_norm,
                interpolation="nearest",
            )
            add_panel_marker(ax, panel_idx, fontsize=5.6)
            ax.set_title(title if row == 1 else "", fontsize=6.4, pad=1.6)
            panel_idx += 1
            apply_map_style(ax, ax is value_axes[0], label if ax is value_axes[0] else None)

        cax = fig.add_subplot(gs[row, n_value])
        cb = fig.colorbar(last_im, cax=cax)
        cb.ax.tick_params(labelsize=5, length=1.5)

        error_axes = [fig.add_subplot(gs[row, n_value + 1 + col]) for col in range(n_error)]
        err_im = None
        for ax, method, array in zip(error_axes, methods, err_grids):
            err_im = ax.imshow(
                masked_for_norm(array, err_norm),
                origin="lower",
                extent=extent,
                cmap="jet",
                norm=err_norm,
                interpolation="nearest",
            )
            add_panel_marker(ax, panel_idx, fontsize=5.6)
            ax.set_title(f"{method.label} error" if row == 1 else "", fontsize=6.4, pad=1.6)
            panel_idx += 1
            apply_map_style(ax, False, None)

        err_cax = fig.add_subplot(gs[row, n_value + 1 + n_error])
        err_cb = fig.colorbar(err_im, cax=err_cax)
        err_cb.ax.tick_params(labelsize=5, length=1.5)

    save_figure(fig, root / output_stem, formats, dpi)
    plt.close(fig)


def draw_all_in_one(
    root: Path,
    methods: list[MethodSpec],
    histories: dict[str, np.ndarray],
    heat_fields: list[str],
    profile_fields: list[str],
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    n_methods = len(methods)
    n_value = 1 + n_methods
    n_error = n_methods
    ncols = n_value + 1 + n_error + 1
    width_ratios = [1.0] * n_value + [0.026] + [1.0] * n_error + [0.026]
    nrows = 1 + len(heat_fields)
    height = max(8.5, 1.05 + 2.95 * len(heat_fields))

    fig = plt.figure(figsize=(26.5, height))
    gs = GridSpec(
        nrows,
        ncols,
        figure=fig,
        height_ratios=[0.66] + [1.55] * len(heat_fields),
        width_ratios=width_ratios,
        hspace=0.18,
        wspace=0.006,
    )

    loss_ax = fig.add_subplot(gs[0, :])
    plot_loss_axis(loss_ax, methods, histories)

    panel_idx = 1
    for row, field in enumerate(heat_fields, start=1):
        true_grid, extent = xyz_to_grid(load_txt(field_file(methods[0], field, "true")))
        pred_grids = [xyz_to_grid(load_txt(field_file(m, field, "pred")))[0] for m in methods]
        err_grids = [xyz_to_grid(load_txt(field_file(m, field, "Maxerror")))[0] for m in methods]

        value_vmin, value_vmax = finite_range([true_grid] + pred_grids)
        value_norm = colors.Normalize(vmin=value_vmin, vmax=value_vmax)
        err_norm = error_norm(err_grids)
        label = FIELD_LABELS.get(field, field.replace("_", " "))

        value_axes = [fig.add_subplot(gs[row, col]) for col in range(n_value)]
        value_titles = ["Exact"] + [m.label for m in methods]
        value_arrays = [true_grid] + pred_grids
        last_im = None
        for ax, title, array in zip(value_axes, value_titles, value_arrays):
            last_im = ax.imshow(
                array,
                origin="lower",
                extent=extent,
                cmap="jet",
                norm=value_norm,
                interpolation="nearest",
            )
            add_panel_marker(ax, panel_idx, fontsize=5.6)
            ax.set_title(title if row == 1 else "", fontsize=6.2, pad=1.5)
            panel_idx += 1
            apply_map_style(ax, ax is value_axes[0], label if ax is value_axes[0] else None)

        cax = fig.add_subplot(gs[row, n_value])
        cb = fig.colorbar(last_im, cax=cax)
        cb.ax.tick_params(labelsize=4.8, length=1.4)

        error_axes = [fig.add_subplot(gs[row, n_value + 1 + col]) for col in range(n_error)]
        err_im = None
        for ax, method, array in zip(error_axes, methods, err_grids):
            err_im = ax.imshow(
                masked_for_norm(array, err_norm),
                origin="lower",
                extent=extent,
                cmap="jet",
                norm=err_norm,
                interpolation="nearest",
            )
            add_panel_marker(ax, panel_idx, fontsize=5.6)
            ax.set_title(f"{method.label} error" if row == 1 else "", fontsize=6.2, pad=1.5)
            panel_idx += 1
            apply_map_style(ax, False, None)

        err_cax = fig.add_subplot(gs[row, n_value + 1 + n_error])
        err_cb = fig.colorbar(err_im, cax=err_cax)
        err_cb.ax.tick_params(labelsize=4.8, length=1.4)

    save_figure(fig, root / "Kerr2D_pdf_style_all_in_one", formats, dpi)
    plt.close(fig)


def draw_profiles(
    root: Path,
    methods: list[MethodSpec],
    histories: dict[str, np.ndarray],
    profile_fields: list[str],
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    if not profile_fields:
        return

    fig = plt.figure(figsize=(25.0, 16.2))
    gs = GridSpec(
        5,
        8,
        figure=fig,
        height_ratios=[1.0, 1.0, 1.0, 0.95, 0.95],
        hspace=0.68,
        wspace=0.34,
    )

    panel_idx = 0
    for row, field in enumerate(profile_fields[:3]):
        ax_val = fig.add_subplot(gs[row, :4])
        ax_err = fig.add_subplot(gs[row, 4:])
        panel_idx = draw_profile_pair(
            root,
            ax_val,
            ax_err,
            methods,
            field,
            panel_idx,
            title_size=8.0,
            tick_size=7.0,
            show_legend=True,
        )

    structure_specs = [
        ("N", 1, 6),
        ("M", 3, 8),
        ("S", 4, 9),
        ("IdentityError", 5, 10),
    ]
    structure_axes = [
        fig.add_subplot(gs[3, 0:2]),
        fig.add_subplot(gs[3, 2:4]),
        fig.add_subplot(gs[3, 4:6]),
        fig.add_subplot(gs[3, 6:8]),
    ]
    for ax, (name, value_col, fdm_col) in zip(structure_axes, structure_specs):
        fdm_ref = None
        for method_index, method in enumerate(methods):
            path = method.out_dir / f"{method.prefix}_structure_convergence.txt"
            data = load_txt(path)
            epoch = data[:, 0]
            y = np.where(data[:, value_col] > 0.0, data[:, value_col], np.nan)
            ax.plot(
                epoch,
                y,
                color=method.color,
                ls=METHOD_LINESTYLES.get(method.label, "-"),
                lw=0.82,
                alpha=0.86,
                label=method.label,
                zorder=3 + method_index,
            )
            if fdm_ref is None:
                fdm_ref = float(data[0, fdm_col])
        if fdm_ref is not None and fdm_ref > 0.0:
            ax.axhline(fdm_ref, color=FDM_COLOR, ls="--", lw=0.95, alpha=0.65, label="FDM", zorder=1)
        add_panel_marker(ax, panel_idx, fontsize=6.2, outside=True)
        panel_idx += 1
        ax.set_title(f"Structure convergence: {name}", fontsize=8, pad=6)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel(name, fontsize=8)
        ax.set_yscale("log")
        ax.grid(True, which="both", lw=0.28, alpha=0.35)
        ax.tick_params(labelsize=7, length=2)
        if name == "N":
            ax.legend(
                fontsize=6,
                frameon=False,
                ncol=3,
                loc="upper left",
                bbox_to_anchor=(0.0, 1.20),
                handlelength=2.0,
            )

    ax_q = fig.add_subplot(gs[4, 0:4])
    ax_q_err = fig.add_subplot(gs[4, 4:8])
    target_q = None
    radius_offsets = np.linspace(-0.045, 0.045, len(methods)) if len(methods) > 1 else [0.0]
    for method_index, method in enumerate(methods):
        path = method.out_dir / f"{method.prefix}_topological_charge.txt"
        data = load_txt(path)
        radius = data[:, 0]
        pred_q = data[:, 2]
        pred_err = np.abs(data[:, 5])
        shifted_radius = radius + radius_offsets[method_index]
        ax_q.plot(
            shifted_radius,
            pred_q,
            color=method.color,
            ls=METHOD_LINESTYLES.get(method.label, "-"),
            marker=METHOD_MARKERS.get(method.label, "o"),
            mec="black",
            mew=0.25,
            ms=4.3,
            lw=0.95,
            label=method.label,
            zorder=3 + method_index,
        )
        ax_q_err.plot(
            shifted_radius,
            pred_err,
            color=method.color,
            ls=METHOD_LINESTYLES.get(method.label, "-"),
            marker=METHOD_MARKERS.get(method.label, "o"),
            mec="black",
            mew=0.25,
            ms=4.3,
            lw=0.95,
            label=method.label,
            zorder=3 + method_index,
        )
        if target_q is None:
            target_q = float(data[0, 3])
    if target_q is not None:
        ax_q.axhline(target_q, color=FDM_COLOR, ls="--", lw=0.95, alpha=0.65, label="target q", zorder=1)

    for ax, title, ylabel in (
        (ax_q, "Topological charge", r"$Q_\gamma$"),
        (ax_q_err, "Topological charge absolute error", "Abs. error"),
    ):
        add_panel_marker(ax, panel_idx, fontsize=6.2, outside=True)
        panel_idx += 1
        ax.set_title(title, fontsize=8, pad=6)
        ax.set_xlabel("Radius", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, lw=0.28, alpha=0.35)
        ax.tick_params(labelsize=7, length=2)
    ax_q_err.set_yscale("symlog", linthresh=1e-16, linscale=0.6)
    ax_q.legend(
        fontsize=6,
        frameon=False,
        ncol=5,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.18),
        handlelength=2.0,
    )
    ax_q_err.legend(
        fontsize=6,
        frameon=False,
        ncol=4,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.18),
        handlelength=2.0,
    )

    save_figure(fig, root / "Kerr2D_pdf_style_profiles", formats, dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redraw Kerr2D LNN-PINN/PINN/PI-VAST/XPINN outputs in the integrated PDF figure style."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root containing method folders.")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI.")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf", "svg"],
        help="Output formats.",
    )
    parser.add_argument(
        "--full-set",
        action="store_true",
        help="Also create legacy integrated-overview and single-field figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    formats = tuple(args.formats)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "savefig.facecolor": "white",
        }
    )

    methods = discover_methods(root, METHOD_ORDER)
    histories = load_histories(methods)
    heat_fields, profile_fields = available_fields(methods)
    if not heat_fields:
        raise RuntimeError("No common 2D fields were found.")

    draw_all_in_one(
        root=root,
        methods=methods,
        histories=histories,
        heat_fields=heat_fields,
        profile_fields=profile_fields,
        formats=formats,
        dpi=args.dpi,
    )

    draw_profiles(root, methods, histories, profile_fields, formats, args.dpi)

    if args.full_set:
        draw_heatmap_figure(
            root=root,
            methods=methods,
            histories=histories,
            fields=heat_fields,
            output_stem="Kerr2D_pdf_style_integrated_overview",
            formats=formats,
            dpi=args.dpi,
        )
        for field in heat_fields:
            draw_heatmap_figure(
                root=root,
                methods=methods,
                histories=histories,
                fields=[field],
                output_stem=f"Kerr2D_pdf_style_{field}",
                formats=formats,
                dpi=args.dpi,
            )

    print("Generated PDF-style figures:")
    generated_stems = ["Kerr2D_pdf_style_all_in_one", "Kerr2D_pdf_style_profiles"]
    if args.full_set:
        generated_stems.append("Kerr2D_pdf_style_integrated_overview")
        generated_stems.extend(f"Kerr2D_pdf_style_{field}" for field in heat_fields)
    for stem in generated_stems:
        for fmt in formats:
            path = root / f"{stem}.{fmt}"
            if path.exists():
                print(path.name)


if __name__ == "__main__":
    main()
