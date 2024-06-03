from typing import Tuple

import cv2
import numpy as np


def get_road_mask(semantic_path, semantic_value: Tuple):
    """

    :param semantic_path: semantic image path
    :param semantic_value: value in (b,g,r) for road pixels
    :return: mask, 0 for invalid, 1 for valid
    """
    semantic_img = cv2.imread(semantic_path)
    mask = cv2.inRange(semantic_img, semantic_value, semantic_value)
    mask = mask / 255
    mask.astype(int)
    return mask

def read_vkitti_png_flow(flow_fn):
    """Convert from .png to (h, w, 2) (flow_x, flow_y) float32 array"""
    # read png to bgr in 16 bit unsigned short

    bgr = cv2.imread(flow_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    h, w, _c = bgr.shape
    assert bgr.dtype == np.uint16 and _c == 3
    # b == invalid flow flag == 0 for sky or other invalid flow
    invalid = bgr[..., 0] == 0
    # g,r == flow_y,x normalized by height,width and scaled to [0;2**16 - 1]
    out_flow = 2.0 / (2 ** 16 - 1.0) * bgr[..., 2:0:-1].astype('f4') - 1
    out_flow[..., 0] *= w - 1
    out_flow[..., 1] *= h - 1
    out_flow[invalid] = 0  # or another value (e.g., np.nan)
    return out_flow, 1 - invalid

# def remove_invalid_pixels(mask, )


def get_extrinsics(ex_path: str, frame_id: int):
    """

    :param ex_path: path to extrinsic.txt
    :param frame_id: number of frame
    :return:
        rot: rotation matrix from  WCS to CCS
        pos: the coordinates of WCS's origin in CCS
    """
    extrinsics = np.loadtxt(ex_path, skiprows=1)[frame_id * 2]
    r11, r12, r13, t1 = extrinsics[2:6]
    r21, r22, r23, t2 = extrinsics[6:10]
    r31, r32, r33, t3 = extrinsics[10:14]
    rot = np.array([[r11, r12, r13],
                    [r21, r22, r23],
                    [r31, r32, r33]])
    pos = np.array([t1, t2, t3])
    return rot, pos


