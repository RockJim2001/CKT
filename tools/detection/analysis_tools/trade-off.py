import os
import matplotlib.pyplot as plt
import numpy as np

all_points = {
    (0.000, 0.000): (90.20, 68.78),
    (0.000, 0.001): (90.15, 66.07),
    (0.000, 0.005): (91.00, 68.47),
    (0.000, 0.025): (90.96, 69.93),
    (0.000, 0.100): (91.08, 71.11),
    (0.000, 0.250): (91.49, 69.15),
    (0.000, 0.500): (91.01, 67.61),
    (0.000, 1.000): (91.39, 66.58),

    (0.001, 0.000): (90.79, 67.34),
    (0.001, 0.025): (90.99, 67.95),
    (0.001, 0.050): (91.30, 72.43),
    (0.001, 0.100): (91.42, 68.70),
    (0.001, 0.250): (91.18, 68.03),

    (0.005, 0.000): (91.15, 67.97),
    (0.005, 0.025): (90.98, 67.14),
    (0.005, 0.100): (91.19, 67.95),
    (0.005, 0.250): (91.17, 65.36),

    (0.025, 0.000): (91.48, 66.97),
    (0.025, 0.025): (91.40, 65.77),
    (0.025, 0.100): (91.13, 65.34),
    (0.025, 0.250): (91.16, 62.96),

    (0.100, 0.000): (90.23, 65.49),
    (0.100, 0.025): (90.44, 64.64),
    (0.100, 0.100): (90.98, 63.12),
    (0.100, 0.250): (90.64, 61.39),

    (0.250, 0.250): (90.46, 57.04),

    (0.500, 0.000): (90.89, 57.42),
    (0.500, 0.025): (90.62, 54.70),
    (0.500, 0.100): (90.30, 54.38),
    (0.500, 0.250): (89.78, 55.29),
}

key_points = {
    "No KD + No DR": (0.000, 0.000),
    "KD-only": (0.025, 0.000),
    "DR-only": (0.000, 0.100),
    "KD + DR": (0.001, 0.050),
}


def get_project_root():
    return os.getcwd()


def draw_on_axis(
    ax,
    all_x,
    all_y,
    key_xy,
    path_x,
    path_y,
    color_all,
    color_line,
    color_key,
    marker_key,
    label_offsets,
):
    ax.scatter(
        all_x, all_y,
        s=52,
        marker="D",
        color="#AF4FA8",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.7,
        zorder=1
    )

    ax.plot(
        path_x, path_y,
        linestyle=(0, (4, 3)),
        linewidth=1.8,
        color=color_line,
        alpha=0.95,
        zorder=2
    )

    for name, (x, y) in key_xy.items():
        size = 250 if name == "KD + DR" else 150
        ax.scatter(
            x, y,
            s=size,
            marker=marker_key[name],
            color=color_key[name],
            edgecolor="#222222",
            linewidth=1.3,
            zorder=4
        )

        dx, dy = label_offsets[name]
        ax.annotate(
            name,
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="center",
            va="bottom" if dy > 0 else "top",
            fontsize=10,
            fontweight="semibold",
            color="#222222",
            zorder=6
        )

    ax.grid(
        True,
        linestyle=(0, (3, 3)),
        linewidth=0.7,
        color="#D7DBE2",
        alpha=0.9
    )
    ax.set_axisbelow(True)

    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#4A4A4A")
        ax.spines[side].set_linewidth(1.0)

    ax.tick_params(
        axis="both",
        which="both",
        top=False,
        right=False,
        labeltop=False,
        labelright=False
    )


def main():
    # ===== easiest-to-tune parameters =====
    FIG_W = 8.5
    FIG_H = 6.2
    SUPYLABEL_X = -0.02

    TOP_YLIM = (62.0, 73.5)
    BOTTOM_YLIM = (54.0, 58.5)

    # 直接手动控制 x 轴范围，最省心
    X_LIM = (89.7, 91.6)
    X_TICKS = [90.0, 90.5, 91.0, 91.5]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.labelweight": "normal",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    color_all = "#C7CDD6"
    color_line = "#7FAFBC"

    color_key = {
        "No KD + No DR": "#4E79A7",
        "KD-only": "#F28E2B",
        "DR-only": "#59A14F",
        "KD + DR": "#E15759",
    }

    marker_key = {
        "No KD + No DR": "o",
        "KD-only": "^",
        "DR-only": "s",
        "KD + DR": (5, 1, 0),
    }

    label_offsets = {
        "No KD + No DR": (0, -12),
        "DR-only": (-5, 16),
        "KD-only": (0, -12),
        "KD + DR": (0, 10),
    }

    all_xy = list(all_points.values())
    all_x = np.array([p[0] for p in all_xy])
    all_y = np.array([p[1] for p in all_xy])

    key_xy = {name: all_points[param] for name, param in key_points.items()}

    path_order = ["No KD + No DR", "DR-only", "KD + DR", "KD-only"]
    path_x = [key_xy[name][0] for name in path_order]
    path_y = [key_xy[name][1] for name in path_order]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1,
        figsize=(FIG_W, FIG_H),
        dpi=1200,
        sharex=False,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.3, 1.2], "hspace": 0.05}
    )
    fig.patch.set_facecolor("white")
    ax_top.set_facecolor("white")
    ax_bottom.set_facecolor("white")

    draw_on_axis(
        ax_top, all_x, all_y, key_xy, path_x, path_y,
        color_all, color_line, color_key, marker_key, label_offsets
    )
    draw_on_axis(
        ax_bottom, all_x, all_y, key_xy, path_x, path_y,
        color_all, color_line, color_key, marker_key, label_offsets
    )

    # ===== broken y-axis =====
    ax_top.set_ylim(*TOP_YLIM)
    ax_bottom.set_ylim(*BOTTOM_YLIM)

    # 直接指定 x 轴，不再自动扩成 89.5~92.0
    ax_top.set_xlim(*X_LIM)

    ax_top.set_yticks(np.arange(62, 74, 2))
    ax_bottom.set_yticks(np.arange(54, 59, 1))
    ax_bottom.set_xticks(X_TICKS)

    # hide touching spines
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False)
    ax_bottom.tick_params(top=False)

    # broken-axis marks
    d = 0.008
    kwargs = dict(
        transform=ax_top.transAxes,
        color="#4A4A4A",
        clip_on=False,
        linewidth=1.0
    )
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # fig.supxlabel("Base mAP", fontsize=12, fontweight="normal")
    # fig.supylabel("Novel mAP", fontsize=12, fontweight="normal", x=SUPYLABEL_X)

    save_dir = os.path.join(get_project_root(), "work_dirs", "figures")
    os.makedirs(save_dir, exist_ok=True)

    png_path = os.path.join(save_dir, "TAR_tradeoff_map_broken_axis.png")
    pdf_path = os.path.join(save_dir, "TAR_tradeoff_map_broken_axis.pdf")
    svg_path = os.path.join(save_dir, "TAR_tradeoff_map_broken_axis.svg")

    fig.savefig(png_path, dpi=1200, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(png_path)
    print(pdf_path)
    print(svg_path)


if __name__ == "__main__":
    main()