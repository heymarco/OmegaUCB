import numpy as np

from approach_names import *
import seaborn as sns

# OMEGA_UCB_1_32 = r"$\omega$-UCB ($\rho = \frac{1}{32}$)"
# OMEGA_UCB_1_16 = r"$\omega$-UCB ($\rho = \frac{1}{16}$)"
# OMEGA_UCB_1_8 = r"$\omega$-UCB ($\rho = \frac{1}{8}$)"
# OMEGA_UCB_1_6 = r"$\omega$-UCB ($\rho = \frac{1}{6}$)"
# OMEGA_UCB_1_5 = r"$\omega$-UCB ($\rho = \frac{1}{5}$)"
# OMEGA_UCB_1_4 = r"$\omega$-UCB ($\rho = \frac{1}{4}$)"
# OMEGA_UCB_1_3 = r"$\omega$-UCB ($\rho = \frac{1}{3}$)"
# OMEGA_UCB_1_2 = r"$\omega$-UCB ($\rho = \frac{1}{2}$)"
# OMEGA_UCB_1 = r"$\omega$-UCB ($\rho = 1$)"
# OMEGA_UCB_2 = r"$\omega$-UCB ($\rho = 2$)"
#
# ETA_UCB_1_32 = r"$\eta$-UCB ($\rho = \frac{1}{32}$)"
# ETA_UCB_1_16 = r"$\eta$-UCB ($\rho = \frac{1}{16}$)"
# ETA_UCB_1_8 = r"$\eta$-UCB ($\rho = \frac{1}{8}$)"
# ETA_UCB_1_6 = r"$\eta$-UCB ($\rho = \frac{1}{6}$)"
# ETA_UCB_1_5 = r"$\eta$-UCB ($\rho = \frac{1}{5}$)"
# ETA_UCB_1_4 = r"$\eta$-UCB ($\rho = \frac{1}{4}$)"
# ETA_UCB_1_3 = r"$\eta$-UCB ($\rho = \frac{1}{3}$)"
# ETA_UCB_1_2 = r"$\eta$-UCB ($\rho = \frac{1}{2}$)"
# ETA_UCB_1 = r"$\eta$-UCB ($\rho = 1$)"
# ETA_UCB_2 = r"$\eta$-UCB ($\rho = 2$)"
#
# MUCB = "m-UCB"
# CUCB = "c-UCB"
# IUCB = "i-UCB"
# BTS = "BTS"
# UCB_SC = "UCB-SC"
# UCB_SC_PLUS = "UCB-SC+"
# BUDGET_UCB = "B-UCB"
# B_GREEDY = "b-greedy"


def color_list():
    final_list = []
    omega_ucbs = [
        OMEGA_UCB_1_32,
        OMEGA_UCB_1_16,
        OMEGA_UCB_1_8,
        OMEGA_UCB_1_4,
        OMEGA_UCB_1_2,
        OMEGA_UCB_1,
        OMEGA_UCB_2
    ]
    eta_ucbs = [
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

    omega_ucb_base_color = "Blues"
    omega_ucb_n_colors = len(omega_ucbs)
    eta_ucb_base_color = "Purples"
    eta_ucb_n_colors = len(eta_ucbs)
    other_ucb_base_color = "Reds"
    other_ucb_n_colors = len(other_ucbs)
    bts_base_color = "Wistia"
    bts_n_colors = len(bts)
    others_colors = "Greens"
    others_n_colors = len(others)

    all_info = [(omega_ucbs, omega_ucb_base_color, omega_ucb_n_colors),
                (eta_ucbs, eta_ucb_base_color, eta_ucb_n_colors),
                (other_ucbs, other_ucb_base_color, other_ucb_n_colors),
                (bts, bts_base_color, bts_n_colors),
                (others, others_colors, others_n_colors)]

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