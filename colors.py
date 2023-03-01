import numpy as np

from approach_names import *
import seaborn as sns


omega_ucb_base_color = "Blues"
other_colors = "Greys"
eta_ucb_base_color = "Wistia"


def color_list():
    final_list = []
    omega_ucbs = [
        OMEGA_UCB_1_64,
        OMEGA_UCB_1_32,
        OMEGA_UCB_1_16,
        OMEGA_UCB_1_8,
        OMEGA_UCB_1_4,
        OMEGA_UCB_1_2,
        OMEGA_UCB_1,
        OMEGA_UCB_2
    ]
    eta_ucbs = [
        ETA_UCB_1_64,
        ETA_UCB_1_32,
        ETA_UCB_1_16,
        ETA_UCB_1_8,
        ETA_UCB_1_4,
        ETA_UCB_1_2,
        ETA_UCB_1,
        ETA_UCB_2
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

    omega_ucb_n_colors = len(omega_ucbs)
    eta_ucb_n_colors = len(eta_ucbs)

    other_n_colors = len(other_ucbs) + len(bts) + len(others)

    all_info = [(omega_ucbs, omega_ucb_base_color, omega_ucb_n_colors),
                (eta_ucbs, eta_ucb_base_color, eta_ucb_n_colors),
                (other_ucbs + bts + others, other_colors, other_n_colors)]

    for approaches, base_color, n_colors in all_info:
        palette = sns.color_palette(base_color, n_colors=n_colors)
        for color, approach in zip(palette, approaches):
            final_list.append((approach, color))

    return final_list


def get_palette_for_approaches(approaches):
    apps_colors = color_list()
    app_color_dict = {app: col for app, col in apps_colors}
    colors = [app_color_dict[approach] for approach in approaches]
    return colors