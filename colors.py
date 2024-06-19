from approach_names import *
import seaborn as sns

omega_ucb_base_color = "RdBu_r"
other_colors = "pastel6"
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
    OMEGA_STAR_UCB_1,
    OMEGA_STAR_UCB_1_4,
    # ETA_UCB_2
]
other_ucbs = [
    UCB_SC_PLUS,
    BUDGET_UCB,
    UCB_B2_name,
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

    chp = sns.color_palette("cubehelix", n_colors=len(final_list), as_cmap=False)
    chp = chp[::-1]
    # chp = sns.cubehelix_palette(n_colors=len(final_list), as_cmap=False)
    return [(key, c) for (key, _), c in zip(final_list, chp)]
    return final_list


def get_palette_for_approaches(approaches):
    apps_colors = color_list()
    app_color_dict = {app: col for app, col in apps_colors}
    colors = {approach: app_color_dict[approach] for approach in approaches}
    return colors


def get_markers_for_approaches(approaches):
    markers = [
        "s", ">", "^", "*", "p", "D",
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


def get_linestyles_for_approaches(approaches):
    longdash = 5
    dash = 3
    dot = 1
    pause = 1
    styles = [
        "",
        (dash, pause),
        (dash, pause, dot, pause),
        (dash, pause, dot, pause, dot, pause),
        (dash, pause, dot, pause, dot, pause, dot, pause),
        (dot, pause),
        (longdash, pause),
        (longdash, pause, dot, pause),
        (longdash, pause, dash, pause),
        (longdash, pause, dash, pause, dot, pause),
        (dot, pause, dash, pause, dot, pause, dot, pause, dash, pause, dash, pause),
        (longdash, pause, dot, pause, dash, pause)
        # (.5, .5),
        # (dash, pause),
        # (dash, pause, 2, 1),
        # (dash, pause, dot, pause, dot, pause),
        # (3, 2, 1, 2, 1, 2),
        # (5, 1),
        # (dash, pause, dot, pause),
        # (5, 2),
        # (dot, pause),
        # (dash, pause, 2, 1),
        # (dash, pause, dot, pause, dot, pause)
    ]
    assert len(styles) >= len(all_approaches)
    full_styles_dict = {
        app: style for app, style in zip(all_approaches, styles)
    }
    relevant_styles = {
        app: full_styles_dict[app] for app in approaches
    }
    return relevant_styles
