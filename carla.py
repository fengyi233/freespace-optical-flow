import os
from math import sin, cos, tan, pi

import cv2
import matplotlib
import matplotlib.transforms as mtransforms
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial.transform import Rotation as R

from pathlib import Path

from utils import flow_to_image

# ====================== initialization ===============================
theta_real = -10  # 0 / 5 / -5 / 10 / -10
pic_num = 70  # 140 for straight-going; 70 for turning

path = Path(f"./data/CARLA/t10_s0_r{theta_real}")

original_img_path = path / f"rgb/00000{pic_num:0>3d}.png"
optical_flow_path = path / f"optical_flow/00000{pic_num:0>3d}.tif"
optical_flow_path_png = path / f"optical_flow/00000{pic_num:0>3d}.png"
semantic_path = path / f"semantic/00000{pic_num:0>3d}.png"
fig_save_path = Path('outputs') / f"figs/00000{pic_num:0>3d}"

v_max = image_h = 480
u_max = image_w = 640
u0 = int(image_w / 2)
v0 = int(image_h / 2)
f = fx = fy = image_w / 2.0
h = 2.4  # height of the camera
l = 2.7  # length of the car
EPS = 1e-9

# plt initialization
# plt.rcParams['font.sans-serif'] = 'Times New Roman'


colors = {
    'road': np.array([128, 64, 128], dtype=np.uint8),
    'lane_line': np.array([50, 234, 157], dtype=np.uint8)
}

def add_right_cax(ax, pad, width):
    # 在一个ax右边追加与之等高的cax.
    # pad是cax与ax的间距,width是cax的宽度.

    axpos = ax.get_position()
    caxpos = mtransforms.Bbox.from_extents(
        axpos.x1 + pad,
        axpos.y0 + 0.017,
        axpos.x1 + pad + width,
        axpos.y1 - 0.017
    )
    cax = ax.figure.add_axes(caxpos)

    return cax


def flow_func(x, vr, theta, delta_f):
    """
    Calculate the optical flow using camera and vehicle parameters.

    Params:
        phi: angle between the vehicle and Z-axis (the vehicle’s turning angle)
        theta: camera rotation angle along the Z-axis
        delta_f: steering angle of the front wheel
        vr: velocity of the rear axle
        vf: velocity of the front axle
        h: camera height
        l: distance between the front and rear axles

        lambda_1 = (u1-u0)/fx*sin(theta)+(v1-v0)/fy*cos(theta)
        lambda_2 = (u1-u0)/fx*cos(theta)-(v1-v0)/fy*sin(theta)
        lambda_3 = lambda_2*h-lambda_1*Xd
        lambda_4 = h-lambda_1*Zd
    """

    v = x[0]
    u = x[1]

    lambda_1 = (u - u0) / fx * sin(theta) + (v - v0) / fy * cos(theta)
    lambda_2 = (u - u0) / fx * cos(theta) - (v - v0) / fy * sin(theta)

    fv_e = -fy * sin(theta) * vr / h * (lambda_1 * lambda_2 - (tan(delta_f) / l) * (1 + lambda_2 ** 2) * h) \
           + fy * cos(theta) * vr / h * (lambda_1 ** 2 - lambda_1 * lambda_2 * h * tan(delta_f) / l)
    fu_e = fx * cos(theta) * vr / h * (lambda_1 * lambda_2 - (tan(delta_f) / l) * (1 + lambda_2 ** 2) * h) \
           + fx * sin(theta) * vr / h * (lambda_1 ** 2 - lambda_1 * lambda_2 * h * tan(delta_f) / l)
    # fu_e = (fu_e - fu_mean) / fu_std
    fv_e = (fv_e - fv_min) / (fv_max - fv_min)
    fu_e = (fu_e - fu_min) / (fu_max - fu_min)
    output = np.hstack((fv_e, fu_e))
    return output


def fv_func(x, vr, theta, delta_f):
    v = x[0]
    u = x[1]

    lambda_1 = (u - u0) / fx * sin(theta) + (v - v0) / fy * cos(theta)
    lambda_2 = (u - u0) / fx * cos(theta) - (v - v0) / fy * sin(theta)

    output = -fy * sin(theta) * vr / h * (lambda_1 * lambda_2 - (tan(delta_f) / l) * (1 + lambda_2 ** 2) * h) \
             + fy * cos(theta) * vr / h * (lambda_1 ** 2 - lambda_1 * lambda_2 * h * tan(delta_f) / l)
    return output.ravel()


def fu_func(x, vr, theta, delta_f):
    v = x[0]
    u = x[1]
    lambda_1 = (u - u0) / fx * sin(theta) + (v - v0) / fy * cos(theta)
    lambda_2 = (u - u0) / fx * cos(theta) - (v - v0) / fy * sin(theta)
    output = fx * cos(theta) * vr / h * (lambda_1 * lambda_2 - (tan(delta_f) / l) * (1 + lambda_2 ** 2) * h) \
             + fx * sin(theta) * vr / h * (lambda_1 ** 2 - lambda_1 * lambda_2 * h * tan(delta_f) / l)
    return output.ravel()


if __name__ == '__main__':
    # ==================== get the mask of freespace =========================================
    semantic_img = cv2.imread(str(semantic_path))
    # road: (128,64,128)
    # road line: (157, 234, 50)
    mask_road = cv2.inRange(semantic_img, colors['road'], colors['road'])
    mask_roadline = cv2.inRange(semantic_img, colors['lane_line'], colors['lane_line'])
    mask = mask_road | mask_roadline
    mask = mask / 255
    mask.astype(int)

    # ==================== get optical flow ground truth =========================================
    flow_img = Image.open(optical_flow_path)
    flow_img = np.reshape(np.array(flow_img), (image_h, image_w, 2))

    fu_origin = flow_img[:, :, 0] * image_w
    fv_origin = -flow_img[:, :, 1] * image_h

    # set 0 for the mask if the optical flow is invalid
    for i in range(v_max):
        for j in range(u_max):
            if fv_origin[i, j] + i > v_max:
                mask[i, j] = 0

    fu = fu_origin * mask
    fv = fv_origin * mask

    if not os.path.exists(fig_save_path):
        os.makedirs(fig_save_path)

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
    vu = np.nonzero(mask)
    fv_val = fv[vu[0], vu[1]]
    fv_val = np.array(fv_val)
    fu_val = fu[vu[0], vu[1]]
    fu_val = np.array(fu_val)
    fv_min = np.min(fv_val)
    fv_max = np.max(fv_val)
    fv_val = (fv_val - fv_min) / (fv_max - fv_min)
    fu_min = np.min(fu_val)
    fu_max = np.max(fu_val)
    fu_val = (fu_val - fu_min) / (fu_max - fu_min)

    flow_val = np.hstack((fv_val, fu_val))

    # fit the f_v
    res = curve_fit(flow_func, vu, flow_val,
                    p0=[1, 5 / 180 * 3.14, 0],
                    bounds=[[0, -0.5 * pi, -0.5 * pi], [5, 0.5 * pi, 0.5 * pi]])
    popt = res[0]

    u = np.arange(u_max)
    v = np.arange(v_max)
    U, V = np.meshgrid(u, v)
    VV = np.expand_dims(V, 0)
    UU = np.expand_dims(U, 0)
    vu_valid = np.append(VV, UU, axis=0)

    # parameter estimation
    vr_est = popt[0]
    theta_est = popt[1]
    delta_f_est = popt[2]

    fu_est = fu_func(vu_valid, vr_est, theta_est, delta_f_est)
    fv_est = fv_func(vu_valid, vr_est, theta_est, delta_f_est)
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

    # AE
    delta_angle = (fu * fu_est + fv * fv_est) / (
            ((fu_est ** 2 + fv_est ** 2) ** 0.5) * ((fu ** 2 + fv ** 2) ** 0.5) + EPS)
    AE = np.arccos(delta_angle)
    AE = np.nan_to_num(AE, nan=0)
    AE_mean = np.sum(AE) / np.sum(mask)

    AEE_R1 = (np.sum(abs(error_L2) > 1)) / np.sum(mask)
    AEE_R2 = (np.sum(abs(error_L2) > 2)) / np.sum(mask)

    theta_est = theta_est * 180 / pi
    delta_f_est = delta_f_est * 180 / pi
    # theta_absolute_error = theta_est - theta_real

    print(f"theta real is {theta_real}, pic number is {pic_num}:")
    print(f"theta_est:   {theta_est:.4f}")
    print(f"theta_err:   {theta_est - theta_real:.4f}")
    print(f"AAE:{AE_mean:.4f}")
    print(f"AEE:{EPE:.4f}")
    print(f"R1:{AEE_R1 * 100:.4f}")
    print(f"R2:{AEE_R2 * 100:.4f}")

    # ==========================================================
    #                     flow ground truth
    # ==========================================================
    flow_gt = np.zeros((fu.shape[0], fu.shape[1], 2))
    flow_gt[:, :, 0] = fu
    flow_gt[:, :, 1] = fv

    flow_img_gt = flow_to_image(flow_gt)
    plt.figure('flow gt')
    plt.imshow(flow_img_gt)
    plt.axis('off')
    plt.savefig(fig_save_path / "flow_gt.png", bbox_inches='tight', pad_inches=0, dpi=500)

    # ==========================================================
    #                     flow est
    # ==========================================================
    flow_est = np.zeros((fu.shape[0], fu.shape[1], 2))
    flow_est[:, :, 0] = fu_est
    flow_est[:, :, 1] = fv_est

    flow_img_est = flow_to_image(flow_est)
    plt.figure('flow est')
    plt.imshow(flow_img_est)
    plt.axis('off')
    plt.savefig(fig_save_path / "flow_estimated.png", bbox_inches='tight', pad_inches=0, dpi=500)

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
    # 设置colorbar标签字体等
    cb.ax.tick_params(labelsize=20, direction='in')
    plt.savefig(fig_save_path / "fv_estimated.png", bbox_inches='tight', pad_inches=0, dpi=500)

    plt.figure('fv_truth')
    ax = plt.axes()
    h1 = plt.imshow(fv, norm=norm, cmap="magma")
    plt.axis('off')
    cax = add_right_cax(ax, pad=0.01, width=0.02)
    # cbar_ax = fig.add_axes(rect)
    cb = plt.colorbar(h1, cax=cax, ticks=None)
    # 设置colorbar标签字体等
    cb.ax.tick_params(labelsize=20, direction='in')  # 设置色标刻度字体大小。
    # cb.ax.set_title('pixels', fontsize=10)

    plt.savefig(fig_save_path / "fv_truth.png", bbox_inches='tight', pad_inches=0, dpi=500)

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
    # cbar_ax = fig.add_axes(rect)
    cb = plt.colorbar(h1, cax=cax, ticks=None)
    # 设置colorbar标签字体等
    cb.ax.tick_params(labelsize=20, direction='in')  # 设置色标刻度字体大小。
    # cb.ax.set_title('pixels', fontsize=7)
    # cb.ax.set_title('pixels',fontsize=5)

    plt.savefig(fig_save_path / "fu_estimated.png", bbox_inches='tight', pad_inches=0, dpi=500)

    plt.figure('fu_truth')
    ax = plt.axes()
    h1 = plt.imshow(fu, norm=norm, cmap="magma")
    plt.axis('off')

    cax = add_right_cax(ax, pad=0.01, width=0.02)
    # cbar_ax = fig.add_axes(rect)
    cb = plt.colorbar(h1, cax=cax, ticks=None)
    # 设置colorbar标签字体等
    cb.ax.tick_params(labelsize=20, direction='in')  # 设置色标刻度字体大小。
    # cb.ax.set_title('pixels', fontsize=7)
    plt.savefig(fig_save_path / "fu_truth.png", bbox_inches='tight', pad_inches=0, dpi=500)

    # ==========================================================
    #                     rgb
    # ==========================================================
    fig = plt.figure("rgb with mask")
    plt.axis('off')
    rgb_img = np.array(Image.open(original_img_path))

    plt.imshow(rgb_img)
    plt.savefig(fig_save_path / "rgb.png", bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)

    # ==========================================================
    #                     semantic
    # ==========================================================
    fig = plt.figure("semantic")
    plt.axis('off')
    plt.imshow(semantic_img)
    plt.savefig(fig_save_path / "semantic.png", bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
    plt.show()
