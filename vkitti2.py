from math import sin, cos, pi
from pathlib import Path

import matplotlib
from scipy.optimize import curve_fit
from scipy.spatial.transform import Rotation as R

from utils import *

# ====================== initialization ===============================
# 4, 200
# scene = 1
# pic_num = 4

u0 = 620.5
v0 = 187
fx = fy = f = 725.0087


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
    args = parse_args()
    pic_num = args.pic_num
    scene = args.scene

    root_path = Path("data/VKITTI2")
    scene_path = f"Scene{scene:0>2d}/clone/frames/"

    flow_path = root_path / "vkitti_2.0.3_forwardFlow/" / scene_path / "forwardFlow/Camera_0/" / f"flow_{pic_num:0>5d}.png"
    semantic_path = root_path / "vkitti_2.0.3_classSegmentation/" / scene_path / "classSegmentation/Camera_0/" / f"classgt_{pic_num:0>5d}.png"
    ex_path = root_path / "vkitti_2.0.3_textgt/" / scene_path.replace('frames/', "") / "extrinsic.txt"

    rgb_path = root_path / "vkitti_2.0.3_rgb/" / scene_path / "rgb/Camera_0/"
    img1_path = rgb_path / f"rgb_{pic_num:0>5d}.jpg"
    img2_path = rgb_path / f"rgb_{pic_num + 1:0>5d}.jpg"

    fig_save_path = Path('../outputs/figs/VKITTI2') / scene_path / f"{pic_num + 1:0>5d}"

    # ==================== obtain freespace mask =========================================
    # 1: valid  0: invalid
    mask = get_road_mask(str(semantic_path), (100, 60, 100))


    # ==================== get camera poses =========================================
    def get_cam_pose(ex_path, pic_num):
        # frame 1
        rotate_mat1, translation1 = get_extrinsics(ex_path, pic_num)
        # frame 2
        rotate_mat2, translation2 = get_extrinsics(ex_path, pic_num + 1)

        # 4x4 homogeneous transformation matrix
        T_world_frame1 = np.vstack([np.hstack([rotate_mat1, translation1.reshape(3, 1)]), np.array([0, 0, 0, 1])])
        T_world_frame2 = np.vstack([np.hstack([rotate_mat2, translation2.reshape(3, 1)]), np.array([0, 0, 0, 1])])
        T_frame2_world = np.linalg.inv(T_world_frame2)  # frame2 in world coordinate
        T_frame2_frame1 = T_world_frame1 @ T_frame2_world  # frame2 in frame1

        p1 = np.linalg.inv(T_world_frame1)[:3, 3]
        p2 = np.linalg.inv(T_world_frame2)[:3, 3]
        print("distance in WCS:", np.linalg.norm(p1 - p2))
        # distance of advance
        x_distance = T_frame2_frame1[0, 3]
        z_distance = T_frame2_frame1[2, 3]
        print(f'x distance of advance (gt): {x_distance:.4f}')
        print(f'z distance of advance (gt): {z_distance:.4f}')

        # r1 = R.from_matrix(rotate_mat1)
        # r2 = R.from_matrix(rotate_mat2)
        r3 = R.from_matrix(T_frame2_frame1[:3, :3])
        # euler1 = r1.as_euler("xyz", degrees=True)
        # euler2 = r2.as_euler("xyz", degrees=True)
        euler3 = r3.as_euler("xyz", degrees=True)

        pitch, yaw, roll = euler3

        # print(f'the euler angle of frame {pic_num} is: {euler1}')
        # print(f'the euler angle of frame {pic_num} is: {euler2}')
        print(f'the euler angle of between {pic_num},{pic_num + 1} is: {euler3}')
        return roll, x_distance, z_distance, yaw


    # ==================== obtain fu fv =========================================
    flow, valid = read_vkitti_png_flow(str(flow_path))
    fu = flow[:, :, 0]
    fv = flow[:, :, 1]

    # plt.figure("fu")
    # plt.imshow(fu)
    # plt.figure("fv")
    # plt.imshow(fv)

    mask[valid == 0] = 0
    v_max = fu.shape[0]
    u_max = fu.shape[1]
    for i in range(v_max):
        for j in range(u_max):
            if fv[i, j] + i > v_max or fu[i, j] + j > u_max or fu[i, j] + j < 0:
                mask[i, j] = 0

    fu = fu * mask
    fv = fv * mask

    vu = np.nonzero(mask * valid)
    fv_val = fv[vu[0], vu[1]]
    fv_val = np.array(fv_val)
    fu_val = fu[vu[0], vu[1]]
    fu_val = np.array(fu_val)

    flow_val = np.hstack((fv_val, fu_val))
    # ==================== fit fv =========================================

    popt, _ = curve_fit(flow_func, vu, flow_val,
                        p0=[0, 0, 0, 0, 1.5],
                        bounds=[[-0.5 * pi, 0, 0, -0.5 * pi, 1], [0.5 * pi, 10, 10, 0.5 * pi, 3]])

    U, V = np.meshgrid(np.arange(u_max), np.arange(v_max))
    VV = np.expand_dims(V, 0)
    UU = np.expand_dims(U, 0)
    vu_valid = np.append(VV, UU, axis=0)

    # parameter estimation
    theta_est = popt[0]
    Xd_est = popt[1]
    Zd_est = popt[2]
    phi_est = popt[3]
    h_est = popt[4]

    roll, x_distance, z_distance, yaw = get_cam_pose(ex_path, pic_num)
    fu_est = fu_func(vu_valid, theta=-np.deg2rad(roll), Xd=x_distance, Zd=z_distance, phi=np.deg2rad(yaw), h=1.5)
    fv_est = fv_func(vu_valid, theta=-np.deg2rad(roll), Xd=x_distance, Zd=z_distance, phi=np.deg2rad(yaw), h=1.5)
    fu_est = fu_est.reshape(v_max, u_max) * mask
    fv_est = fv_est.reshape(v_max, u_max) * mask

    fu_diff = fu_est - fu
    fv_diff = fv_est - fv

    fv_diff_mean = np.sum(abs(fv_diff)) / (np.sum(mask))
    fu_diff_mean = np.sum(abs(fu_diff)) / (np.sum(mask))
    print('========================================')
    print(f'theta_est:      {theta_est:.4f}')
    print(f'Xd_est:         {Xd_est:.4f}')
    print(f'Zd_est:         {Zd_est:.4f}')
    print(f'phi_est:        {phi_est:.4f}')
    print(f'h_est:          {h_est:.4f}')
    print(f'avg fu err:     {fu_diff_mean:.4f}')
    print(f'avg fv err:     {fv_diff_mean:.4f}')
    print('========================================')

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
    # cbar_ax = fig.add_axes(rect)
    cb = plt.colorbar(h1, cax=cax, ticks=None)
    cb.ax.tick_params(labelsize=10, direction='in')
    # plt.savefig(fig_save_path / "/fv_estimation.png", bbox_inches='tight', pad_inches=0, dpi=500)

    plt.figure('fv_truth')
    ax = plt.axes()
    h1 = plt.imshow(fv, norm=norm, cmap="magma")
    plt.axis('off')

    cax = add_right_cax(ax, pad=0.01, width=0.02)
    # cbar_ax = fig.add_axes(rect)
    cb = plt.colorbar(h1, cax=cax, ticks=None)
    cb.ax.tick_params(labelsize=10, direction='in')
    # cb.ax.set_title('pixels', fontsize=10)

    # plt.savefig(fig_save_path / "/fv_truth.png", bbox_inches='tight', pad_inches=0, dpi=500)

    # norm = matplotlib.colors.Normalize(vmin=-2, vmax=2)
    # norm = matplotlib.colors.Normalize(vmin=-2, vmax=2)

    # plt.figure('fv_absolute')
    # ax = plt.axes()
    # # h1 = plt.imshow(fv_diff, norm=norm,cmap="magma")
    # h1 = plt.imshow(abs(fv_diff), cmap="magma")
    # plt.axis('off')
    # cax = add_right_cax(ax, pad=0.01, width=0.02)
    # cb = plt.colorbar(h1, cax=cax, ticks=None)
    # cb.ax.tick_params(labelsize=10, direction='in')
    # cb.ax.set_title('pixels', fontsize=7)
    # plt.savefig(fig_save_path / "/fv_absolute.png", bbox_inches='tight', pad_inches=0, dpi=500)

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
    cb.ax.tick_params(labelsize=10, direction='in')
    # cb.ax.set_title('pixels', fontsize=7)
    # cb.ax.set_title('pixels',fontsize=5)

    # plt.savefig(fig_save_path / "/fu_estimation.png", bbox_inches='tight', pad_inches=0, dpi=500)

    plt.figure('fu_truth')
    ax = plt.axes()
    h1 = plt.imshow(fu, norm=norm, cmap="magma")
    plt.axis('off')

    cax = add_right_cax(ax, pad=0.01, width=0.02)
    # cbar_ax = fig.add_axes(rect)
    cb = plt.colorbar(h1, cax=cax, ticks=None)
    cb.ax.tick_params(labelsize=10, direction='in')
    # cb.ax.set_title('pixels', fontsize=7)
    # plt.savefig(fig_save_path / "/fu_truth.png", bbox_inches='tight', pad_inches=0, dpi=500)

    # norm = matplotlib.colors.Normalize(vmin=-2, vmax=2)
    # norm = matplotlib.colors.Normalize(vmin=-2, vmax=2)

    # plt.figure('fu_absolute')
    # ax = plt.axes()
    # h1 = plt.imshow(abs(fu_diff), cmap="magma")
    # # h1 = plt.imshow(fu_diff, norm=norm,cmap="magma")
    # plt.axis('off')
    # cax = add_right_cax(ax, pad=0.01, width=0.02)
    # cb = plt.colorbar(h1, cax=cax, ticks=None)
    # cb.ax.tick_params(labelsize=10, direction='in')
    # cb.ax.set_title('pixels', fontsize=7)
    # plt.savefig(fig_save_path / "/fu_absolute.png", bbox_inches='tight', pad_inches=0, dpi=500)

    # ==========================================================
    #                     rgb
    # ==========================================================
    semantic_img = cv2.imread(str(semantic_path))

    mask = cv2.inRange(semantic_img, (100, 60, 100), (100, 60, 100))
    mask = mask / 255
    mask.astype(int)

    fig = plt.figure("rgb with mask")
    plt.axis('off')
    rgb_img = cv2.imread(str(img1_path), -1)[..., ::-1]
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
    # plt.savefig(fig_save_path / "/mask.png", bbox_inches='tight', dpi=500, pad_inches=0.0)
    plt.show()
