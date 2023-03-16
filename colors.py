import numpy as np

from approach_names import *
import seaborn as sns


omega_ucb_base_color = "RdBu_r"
other_colors = "Greys"
eta_ucb_base_color = "RdBu"

omega_ucbs = [
    # OMEGA_UCB_1_64,
    # OMEGA_UCB_1_32,
    # OMEGA_UCB_1_16,
    # OMEGA_UCB_1_8,
    # OMEGA_UCB_1_2,
    OMEGA_UCB_1,
    OMEGA_UCB_1_4,
    # OMEGA_UCB_2
]
eta_ucbs = [
    # ETA_UCB_1_64,
    # ETA_UCB_1_32,
    # ETA_UCB_1_16,
    # ETA_UCB_1_8,
    # ETA_UCB_1_2,
    ETA_UCB_1,
    ETA_UCB_1_4,
    # ETA_UCB_2
]
other_ucbs = [
    UCB_SC_PLUS,
    BUDGET_UCB,
    MUCB,
    CUCB,
    IUCB
]
bts = [
    BTS
]
others = [
    B_GREEDY
]

all_approaches = omega_ucbs + eta_ucbs + other_ucbs + bts + others

omega_palette = sns.color_palette(omega_ucb_base_color, n_colors=max(len(omega_ucbs) * 5, 6))[1:]
eta_palette = sns.color_palette(eta_ucb_base_color, n_colors=max(len(eta_ucbs) * 5, 6))[1:]
others_palette = sns.color_palette(other_colors, n_colors=len(other_ucbs) + len(bts) + len(others) + 1)[1:]


def color_list():
    final_list = []

    omega_ucb_n_colors = len(omega_ucbs)
    eta_ucb_n_colors = len(eta_ucbs)

    other_n_colors = len(other_ucbs) + len(bts) + len(others)

    all_info = [(omega_ucbs, omega_palette, omega_ucb_n_colors),
                (eta_ucbs, eta_palette, eta_ucb_n_colors),
                (other_ucbs + bts + others, others_palette, other_n_colors)]

    for approaches, palette, n_colors in all_info:
        for color, approach in zip(palette[:len(approaches)], approaches):
            final_list.append((approach, color))

    return final_list


def get_palette_for_approaches(approaches):
    apps_colors = color_list()
    app_color_dict = {app: col for app, col in apps_colors}
    colors = [app_color_dict[approach] for approach in approaches]
    colors = {approach: app_color_dict[approach] for approach in approaches}
    return colors


def get_markers_for_approaches(approaches):
    markers = [
        "s", ">", "^", "*", "p",
        "v", "P", "d", "X", "H", "<"
    ]
    assert len(markers) >= len(all_approaches)
    full_marker_dict = {
        app: marker for app, marker in zip(all_approaches, markers)
    }
    relevant_markers = {
        app: full_marker_dict[app] for app in approaches
    }
    return relevant_markers
