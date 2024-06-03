import os
from math import sin, cos, pi
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from utils import add_right_cax

# ====================== initialization ===============================
pic_num_list = [1, 2, 26, 69, 70, 71, 72, 76, 80, 99, 101, 102, 106, 128, 147, 148, 176, 188, 194, 195, 197]

# selected_pic = [72,99,148,197]
pic_num = 72

u0 = 609.5593
v0 = 172.8540
fx = fy = f = 721.5377
EPS = 1e-9

path = Path("../data/KITTI/")
img_path = path / f"image_2/000{pic_num:0>3d}_10.png"
optical_flow_path = path / f"flow_noc/000{pic_num:0>3d}_10.png"
semantic_path = path / f"semantic/000{pic_num:0>3d}_10.png"
fig_save_path = Path('../outputs') / f"figs/KITTI/000{pic_num:0>3d}"
if not os.path.exists(fig_save_path):
    os.makedirs(fig_save_path)


def flow_func(x, theta, Xd, Zd, phi, h):
    v = x[0]
    u = x[1]

    lambda_1 = (u - u0) / fx * sin(theta) + (v - v0) / fy * cos(theta)
    lambda_2 = (u - u0) / fx * cos(theta) - (v - v0) / fy * sin(theta)
    lambda_3 = lambda_2 * h - lambda_1 * Xd
    lambda_4 = h - lambda_1 * Zd

    lambda_5 = (lambda_3 * cos(phi) - lambda_4 * sin(phi)) / \
               (lambda_3 * sin(phi) + lambda_4 * cos(phi)) - lambda_2
    lambda_6 = (lambda_1 * h) / (lambda_3 * sin(phi) + lambda_4 * cos(phi)) - lambda_1

    fv_e = fy * (-lambda_5 * sin(theta) + lambda_6 * cos(theta))
    fu_e = fx * (lambda_5 * cos(theta) + lambda_6 * sin(theta))

    output = np.hstack((fv_e, fu_e))
    return output


def fv_func(x, theta, Xd, Zd, phi, h):
    v = x[0]
    u = x[1]

    lambda_1 = (u - u0) / fx * sin(theta) + (v - v0) / fy * cos(theta)
    lambda_2 = (u - u0) / fx * cos(theta) - (v - v0) / fy * sin(theta)
    lambda_3 = lambda_2 * h - lambda_1 * Xd
    lambda_4 = h - lambda_1 * Zd

    lambda_5 = (lambda_3 * cos(phi) - lambda_4 * sin(phi)) / \
               (lambda_3 * sin(phi) + lambda_4 * cos(phi)) - lambda_2
    lambda_6 = (lambda_1 * h) / (lambda_3 * sin(phi) + lambda_4 * cos(phi)) - lambda_1

    output = fy * (-lambda_5 * sin(theta) + lambda_6 * cos(theta))

    return output


def fu_func(x, theta, Xd, Zd, phi, h):
    v = x[0]
    u = x[1]

    lambda_1 = (u - u0) / fx * sin(theta) + (v - v0) / fy * cos(theta)
    lambda_2 = (u - u0) / fx * cos(theta) - (v - v0) / fy * sin(theta)
    lambda_3 = lambda_2 * h - lambda_1 * Xd
    lambda_4 = h - lambda_1 * Zd

    lambda_5 = (lambda_3 * cos(phi) - lambda_4 * sin(phi)) / \
               (lambda_3 * sin(phi) + lambda_4 * cos(phi)) - lambda_2
    lambda_6 = (lambda_1 * h) / (lambda_3 * sin(phi) + lambda_4 * cos(phi)) - lambda_1

    output = fx * (lambda_5 * cos(theta) + lambda_6 * sin(theta))

    return output


if __name__ == '__main__':
    # ==================== get the mask of freespace =========================================
    semantic_img = cv2.imread(str(semantic_path))

    mask = cv2.inRange(semantic_img, (0, 0, 255), (0, 0, 255))
    mask = mask / 255
    mask.astype(int)

    # ==================== get optical flow ground truth =========================================
    optical_flow_img = cv2.imread(str(optical_flow_path), -1)
    valid = optical_flow_img[:, :, 0]
    fu = optical_flow_img[:, :, 2].astype(np.float32)
    fv = optical_flow_img[:, :, 1].astype(np.float32)
    fu = (fu - 2 ** 15) / 64.0
    fv = (fv - 2 ** 15) / 64.0

    mask[valid == 0] = 0

    v_max = fu.shape[0]
    u_max = fu.shape[1]
    for i in range(v_max):
        for j in range(u_max):
            if fv[i, j] + i > v_max or fu[i, j] + j > u_max or fu[i, j] + j < 0:
                mask[i, j] = 0

    fu = fu * mask
    fv = fv * mask

    # ==================== v-fv map =========================================
    # This figure describes the relationship between vertical coordinates `v` and vertical components of optical flow
    # `fv`. It can be seen from the figure that there is a functional relationship between these two variables.
    v_fv_map = np.zeros((v_max, round(fv.max()) + 1))
    for i in range(v_max):
        for j in range(u_max):
            if round(fv[i, j]) != 0:
                v_fv_map[i, round(fv[i, j])] += 1

    plt.figure("v-fv-map")
    plt.title(r"$v-f_v$ map")
    plt.imshow(v_fv_map)
    plt.xlabel(r'$f_v$')
    plt.ylabel(r'$v$')
    plt.savefig(fig_save_path / "v-fv-map.png", bbox_inches='tight', pad_inches=0, dpi=500)

    # ==================== v-fv curve =========================================
    # In this part, we obtain the v-fv curve based on the deduced relationship (Eq. 23) in the paper. The relationship
    # goes as follows:
    # fv = (v-v_0)^2 / (h * f_y / z_d - (v-v_0))

    # select non-zero points
    v_head = v0
    v_tail = v_max
    for i in np.arange(round(v0) + 1, v_max):
        if np.argmax(v_fv_map[i, :]) != 0:
            v_head = i
            break
    for i in np.arange(v_max - 1, v_head, -1):
        if np.argmax(v_fv_map[i, :]) != 0:
            v_tail = i + 1
            break
    v1 = np.arange(v_head + 10, v_tail - 10, 0.1)


    def fitting_curve_s(x, var1, var2):
        output = (x - var2) ** 2 / (var1 - (x - var2))
        return output


    res = curve_fit(fitting_curve_s,
                    np.arange(250, v_tail),
                    np.argmax(v_fv_map, 1)[250:v_tail],
                    p0=[1000, 200],
                    bounds=(np.zeros(2), np.array([5000, 600]))
                    )
    popt = res[0]

    Y = (v1 - popt[1]) ** 2 / (popt[0] - (v1 - popt[1]))
    plt.plot(Y, v1, 'r')
    plt.savefig(fig_save_path / "v-fv-curve.png", bbox_inches='tight', pad_inches=0, dpi=500)

    # ==================== verification of the complete formula =========================================
    # use points of the valid mask
    vu = np.nonzero(mask * valid)
    fv_val = fv[vu[0], vu[1]]
    fv_val = np.array(fv_val)
    fu_val = fu[vu[0], vu[1]]
    fu_val = np.array(fu_val)

    flow_val = np.hstack((fv_val, fu_val))
    res = curve_fit(flow_func, vu, flow_val,
                    p0=[0, 0, 0, 0, 1.5],
                    bounds=[[-0.5 * pi, 0, 0, -0.5 * pi, 1], [0.5 * pi, 10, 10, 0.5 * pi, 3]])
    popt = res[0]

    u = np.arange(u_max)
    v = np.arange(v_max)
    U, V = np.meshgrid(u, v)
    VV = np.expand_dims(V, 0)
    UU = np.expand_dims(U, 0)
    vu_valid = np.append(VV, UU, axis=0)

    # parameter estimation
    theta_est = popt[0]
    Xd_est = popt[1]
    Zd_est = popt[2]
    phi_est = popt[3]
    h_est = popt[4]

    fu_est = fu_func(vu_valid, theta_est, Xd_est, Zd_est, phi_est, h_est)
    fv_est = fv_func(vu_valid, theta_est, Xd_est, Zd_est, phi_est, h_est)
    fu_est = fu_est.reshape(v_max, u_max) * mask
    fv_est = fv_est.reshape(v_max, u_max) * mask
    # absolute error
    fu_diff = fu_est - fu
    fv_diff = fv_est - fv

    fv_diff_mean = np.sum(abs(fv_diff)) / (np.sum(mask))
    fu_diff_mean = np.sum(abs(fu_diff)) / (np.sum(mask))

    # EPE
    error_L2 = (fv_diff ** 2 + fu_diff ** 2) ** 0.5
    EPE = np.sum(error_L2) / np.sum(mask)

    print(f"Average fv Err:         {fv_diff_mean:.4f} (pixels)")
    print(f"Average fu Err:         {fu_diff_mean:.4f} (pixels)")
    print(f"Average Endpoint Err:   {EPE:.4f}")

    # ==========================================================
    #                     fv figure
    # ==========================================================
    plt.rc('font', family='Times New Roman')
    norm = matplotlib.colors.Normalize(vmin=min(fv.min(), fv_est.min()),
                                       vmax=max(fv.max(), fv_est.max()))

    # plot fv
    plt.figure('fv_estimation')
    ax = plt.axes()
    h1 = plt.imshow(fv_est, norm=norm, cmap="magma")
    plt.axis('off')

    cax = add_right_cax(ax, pad=0.01, width=0.02)
    cb = plt.colorbar(h1, cax=cax, ticks=None)
    cb.ax.tick_params(labelsize=20, direction='in')
    plt.savefig(fig_save_path / "fv_estimated.png", bbox_inches='tight', pad_inches=0, dpi=500)

    plt.figure('fv_truth')
    ax = plt.axes()
    h1 = plt.imshow(fv, norm=norm, cmap="magma")
    plt.axis('off')
    cax = add_right_cax(ax, pad=0.01, width=0.02)
    # cbar_ax = fig.add_axes(rect)
    cb = plt.colorbar(h1, cax=cax, ticks=None)
    cb.ax.tick_params(labelsize=20, direction='in')
    # cb.ax.set_title('pixels', fontsize=10)

    plt.savefig(fig_save_path / "fv_truth.png", bbox_inches='tight', pad_inches=0, dpi=500)

    plt.figure('fv_absolute')
    ax = plt.axes()
    # h1 = plt.imshow(fv_diff, norm=norm,cmap="magma")
    h1 = plt.imshow(abs(fv_diff), cmap="magma")
    plt.axis('off')
    cax = add_right_cax(ax, pad=0.01, width=0.02)
    cb = plt.colorbar(h1, cax=cax, ticks=None)
    cb.ax.tick_params(labelsize=10, direction='in')
    # cb.ax.set_title('pixels', fontsize=7)
    plt.savefig(fig_save_path / "fv_absolute.png", bbox_inches='tight', pad_inches=0, dpi=500)

    # ==========================================================
    #                     fu figure
    # ==========================================================

    norm = matplotlib.colors.Normalize(vmin=min(fu.min(), fu_est.min()),
                                       vmax=max(fu.max(), fu_est.max()))
    plt.figure('fu_estimation')
    ax = plt.axes()
    h1 = plt.imshow(fu_est, norm=norm, cmap="magma")
    plt.axis('off')

    cax = add_right_cax(ax, pad=0.01, width=0.02)
    cb = plt.colorbar(h1, cax=cax, ticks=None)
    cb.ax.tick_params(labelsize=20, direction='in')

    plt.savefig(fig_save_path / "fu_estimated.png", bbox_inches='tight', pad_inches=0, dpi=500)

    plt.figure('fu_truth')
    ax = plt.axes()
    h1 = plt.imshow(fu, norm=norm, cmap="magma")
    plt.axis('off')

    cax = add_right_cax(ax, pad=0.01, width=0.02)
    cb = plt.colorbar(h1, cax=cax, ticks=None)
    cb.ax.tick_params(labelsize=20, direction='in')
    # cb.ax.set_title('pixels', fontsize=7)
    plt.savefig(fig_save_path / "fu_truth.png", bbox_inches='tight', pad_inches=0, dpi=500)

    plt.figure('fu_absolute')
    ax = plt.axes()
    h1 = plt.imshow(abs(fu_diff), cmap="magma")
    # h1 = plt.imshow(fu_diff, norm=norm,cmap="magma")
    plt.axis('off')
    cax = add_right_cax(ax, pad=0.01, width=0.02)
    cb = plt.colorbar(h1, cax=cax, ticks=None)
    cb.ax.tick_params(labelsize=10, direction='in')
    # cb.ax.set_title('pixels', fontsize=7)
    plt.savefig(fig_save_path / "fu_absolute.png", bbox_inches='tight', pad_inches=0, dpi=500)

    semantic_img = cv2.imread(str(semantic_path))
    # road: (0, 0, 255)
    mask = cv2.inRange(semantic_img, (0, 0, 255), (0, 0, 255))
    mask = mask / 255
    mask.astype(int)

    # ==========================================================
    #                     rgb
    # ==========================================================
    fig = plt.figure("rgb with mask")
    plt.axis('off')
    rgb_img = np.array(Image.open(img_path))
    # mask = mask * 80
    mask = mask * 50
    mask = mask.astype(np.uint8)
    for i in np.arange(rgb_img.shape[0]):
        for j in np.arange(rgb_img.shape[1]):
            if rgb_img[i, j, 1] <= 255 - mask[i, j]:
                rgb_img[i, j, 1] += mask[i, j]
            else:
                rgb_img[i, j, 1] = 255

    plt.imshow(rgb_img)
    plt.savefig(fig_save_path / "semantic.png", bbox_inches='tight', dpi=500, pad_inches=0.0)
    plt.show()
