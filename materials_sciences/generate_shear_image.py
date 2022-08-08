# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 11:07:16 2021

@author: in7576
"""

import os
import numpy as np
import matplotlib.image as mpli
import scipy.interpolate as spi


def save_tool_greyscale_image(image_path, data_path, el_coords_xyz, gamma):
    """
    Plots an image of the shear-strain from mesh nodes.
    """

    x_min = -150.0
    x_max = 150.0
    num_pxl_x = 300

    y_min = -230.0
    y_max = 230.0
    num_pxl_y = 520

    h_min = 0
    h_max = 90.0

    x_grid = np.linspace(x_min, x_max, num_pxl_x)
    y_grid = np.linspace(y_min, y_max, num_pxl_y)
    pix_x, pix_y = np.meshgrid(x_grid, y_grid)

    nodes_xy = el_coords_xyz[:2, :].T
    pix_gs = spi.griddata(nodes_xy, gamma, (pix_x, pix_y), method='linear')
    pix_gs = np.flipud(pix_gs)

    np.save(data_path, pix_gs)
    mpli.imsave(image_path, pix_gs, cmap='gray', vmin=h_min, vmax=h_max)


if __name__ == '__main__':
    # ======================================
    # load element-coordinates and according shear-data:
    f_dir = os.path.dirname('__file__')
    all_sim_runs_gamma = np.load(os.path.join(f_dir, "piece", 'y_vals_short.npy'))
    el_coords_xyz = np.load(os.path.join(f_dir, "piece", 'el_coords_xyz.npy'))

    sim_data_x = np.load(os.path.join(f_dir, "sim_data", "x_vals.npy"))
    sim_data_y = np.load(os.path.join(f_dir, "sim_data", "y_vals.npy"))

    sel_sim_run_gamma = all_sim_runs_gamma[1, :]  # select one specific simulation run
    abs_path_out_image = os.path.join(f_dir, "images", 'shear.png')
    abs_path_out_data = os.path.join(f_dir, "data", "shear.npy")
    save_tool_greyscale_image(abs_path_out_image, abs_path_out_data, el_coords_xyz, sel_sim_run_gamma)
