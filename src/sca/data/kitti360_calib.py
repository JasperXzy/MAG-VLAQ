"""
KITTI-360 calibration loading and 3D-to-2D projection utilities.

Supports projecting velodyne 3D points onto fisheye camera images (cam 02/03)
to obtain per-point RGB colors for Utonia's 9-channel input (XYZ+RGB+Normal).

Requires KITTI-360 calibration files in {dataroot}/calibration/:
  - calib_cam_to_velo.txt
  - calib_cam_to_pose.txt
  - image_02.yaml  (fisheye intrinsics)

Download from: https://www.cvlibs.net/datasets/kitti-360/download.php
"""

import os
import re
import logging
import numpy as np
from PIL import Image


def _read_variable(fid, name, M, N):
    """Read a named variable from a calibration file."""
    fid.seek(0)
    for line in fid:
        line = line.strip()
        if line.startswith(name + ':'):
            data = line.split(':')[1].strip().split()
            return np.array([float(x) for x in data]).reshape(M, N)
    raise ValueError(f"Variable '{name}' not found in calibration file")


def _read_yaml_file(filepath):
    """Read an OpenCV YAML file (KITTI-360 format) into a dict."""
    import yaml
    with open(filepath, 'r') as f:
        content = f.read()
    # Make OpenCV YAML compatible with Python YAML parser
    # Remove %YAML header
    content = re.sub(r'%YAML.*\n', '', content)
    content = re.sub(r'---\n', '', content)
    # Add space after colon where missing (OpenCV YAML quirk)
    content = re.sub(r':([^ \n])', r': \1', content)
    return yaml.safe_load(content)


def _build_fisheye_dict(T_velo_to_cam, fi):
    """Package one fisheye camera's parameters into a flat dict."""
    return {
        'T_velo_to_cam': T_velo_to_cam,
        'xi': fi['mirror_parameters']['xi'],
        'k1': fi['distortion_parameters']['k1'],
        'k2': fi['distortion_parameters']['k2'],
        'gamma1': fi['projection_parameters']['gamma1'],
        'gamma2': fi['projection_parameters']['gamma2'],
        'u0': fi['projection_parameters']['u0'],
        'v0': fi['projection_parameters']['v0'],
        'orig_width': fi['image_width'],
        'orig_height': fi['image_height'],
    }


def load_calibration(calib_dir):
    """Load KITTI-360 calibration with BOTH fisheye cameras (image_02 + image_03).

    Returns dict:
        {
            'cam02': {...fisheye params + T_velo_to_cam...},
            'cam03': {...fisheye params + T_velo_to_cam...},
        }
    """
    calib_cam_to_velo_path = os.path.join(calib_dir, 'calib_cam_to_velo.txt')
    calib_cam_to_pose_path = os.path.join(calib_dir, 'calib_cam_to_pose.txt')
    fisheye02_path = os.path.join(calib_dir, 'image_02.yaml')
    fisheye03_path = os.path.join(calib_dir, 'image_03.yaml')

    for p in [calib_cam_to_velo_path, calib_cam_to_pose_path, fisheye02_path, fisheye03_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Calibration file not found: {p}\n"
                "Download KITTI-360 calibration from: "
                "https://www.cvlibs.net/datasets/kitti-360/download.php"
            )

    lastrow = np.array([[0, 0, 0, 1]])
    T_cam0_to_velo = np.concatenate(
        (np.loadtxt(calib_cam_to_velo_path).reshape(3, 4), lastrow)
    )

    with open(calib_cam_to_pose_path, 'r') as fid:
        T_cam0_to_pose = np.concatenate((_read_variable(fid, 'image_00', 3, 4), lastrow))
        T_cam2_to_pose = np.concatenate((_read_variable(fid, 'image_02', 3, 4), lastrow))
        T_cam3_to_pose = np.concatenate((_read_variable(fid, 'image_03', 3, 4), lastrow))

    # Chain: velo -> cam0 -> pose -> camX
    T_velo_to_cam0 = np.linalg.inv(T_cam0_to_velo)
    T_velo_to_cam2 = np.linalg.inv(T_cam2_to_pose) @ T_cam0_to_pose @ T_velo_to_cam0
    T_velo_to_cam3 = np.linalg.inv(T_cam3_to_pose) @ T_cam0_to_pose @ T_velo_to_cam0

    fi02 = _read_yaml_file(fisheye02_path)
    fi03 = _read_yaml_file(fisheye03_path)

    return {
        'cam02': _build_fisheye_dict(T_velo_to_cam2, fi02),
        'cam03': _build_fisheye_dict(T_velo_to_cam3, fi03),
    }


def project_velo_to_fisheye(points, fisheye, img_w, img_h):
    """Project velodyne 3D points to one fisheye camera (MEI omnidirectional model).

    Args:
        points: [N, 3] numpy array of velodyne XYZ coordinates
        fisheye: single-camera dict (e.g. calib['cam02'])
        img_w, img_h: actual (resized) image dimensions

    Returns:
        u, v: [N] float pixel coordinates in the resized image
        valid: [N] boolean mask for points with valid projection
    """
    N = points.shape[0]

    pts_hom = np.concatenate([points, np.ones((N, 1))], axis=1)  # [N, 4]
    pts_cam = (fisheye['T_velo_to_cam'] @ pts_hom.T).T[:, :3]   # [N, 3]

    depth = pts_cam[:, 2]
    valid = depth > 0.1

    norm = np.linalg.norm(pts_cam, axis=1)
    norm = np.clip(norm, 1e-8, None)

    x = pts_cam[:, 0] / norm
    y = pts_cam[:, 1] / norm
    z = pts_cam[:, 2] / norm

    xi = fisheye['xi']
    denom = z + xi
    denom = np.clip(np.abs(denom), 1e-8, None) * np.sign(denom + 1e-12)
    x = x / denom
    y = y / denom

    k1, k2 = fisheye['k1'], fisheye['k2']
    ro2 = x * x + y * y
    distort = 1 + k1 * ro2 + k2 * ro2 * ro2
    x = x * distort
    y = y * distort

    u_orig = fisheye['gamma1'] * x + fisheye['u0']
    v_orig = fisheye['gamma2'] * y + fisheye['v0']

    scale_x = img_w / fisheye['orig_width']
    scale_y = img_h / fisheye['orig_height']
    u = u_orig * scale_x
    v = v_orig * scale_y

    valid = valid & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    return u, v, valid


def _colorize_single(points, image, fisheye):
    """Project points into one fisheye image and return (rgb, valid)."""
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))
    img_h, img_w = image.shape[:2]
    u, v, valid = project_velo_to_fisheye(points, fisheye, img_w, img_h)

    rgb = np.zeros((points.shape[0], 3), dtype=np.float32)
    if valid.any():
        u_int = np.clip(u[valid].astype(np.int32), 0, img_w - 1)
        v_int = np.clip(v[valid].astype(np.int32), 0, img_h - 1)
        rgb[valid] = image[v_int, u_int, :3].astype(np.float32) / 255.0
    return rgb, valid


def colorize_points(points, img02, img03, calib):
    """Multi-view per-point RGB: project to BOTH fisheye cameras (image_02 and image_03)
    and merge. Matches the Utonia paper convention:
      "project image colors onto points using the provided calibration,
       and assign signal-black to points not visible in any view."

    Args:
        points: [N, 3] velodyne XYZ
        img02: PIL.Image or np.ndarray for the left fisheye, or None
        img03: PIL.Image or np.ndarray for the right fisheye, or None
        calib: dict with keys 'cam02', 'cam03' from load_calibration()

    Returns:
        rgb: [N, 3] float32 in [0,1]. Points seen by neither camera → [0,0,0].
             Points seen by both cameras → average of the two samples.
    """
    N = points.shape[0]
    rgb_sum = np.zeros((N, 3), dtype=np.float32)
    count = np.zeros((N,), dtype=np.float32)

    if img02 is not None and 'cam02' in calib:
        rgb2, valid2 = _colorize_single(points, img02, calib['cam02'])
        rgb_sum[valid2] += rgb2[valid2]
        count[valid2] += 1.0

    if img03 is not None and 'cam03' in calib:
        rgb3, valid3 = _colorize_single(points, img03, calib['cam03'])
        rgb_sum[valid3] += rgb3[valid3]
        count[valid3] += 1.0

    rgb = np.zeros((N, 3), dtype=np.float32)
    seen = count > 0
    rgb[seen] = rgb_sum[seen] / count[seen, None]
    return rgb


# Module-level cache for calibration (loaded once per dataroot)
_calib_cache = {}


def get_calibration(dataroot):
    """Get cached calibration for a dataroot. Returns None if calibration files missing."""
    if dataroot not in _calib_cache:
        calib_dir = os.path.join(dataroot, 'calibration')
        try:
            _calib_cache[dataroot] = load_calibration(calib_dir)
        except FileNotFoundError as e:
            logging.warning(
                f"KITTI-360 calibration not found: {e}\n"
                "RGB will be set to zeros. To enable color projection, "
                "download calibration from https://www.cvlibs.net/datasets/kitti-360/download.php "
                f"and place files in {calib_dir}/"
            )
            _calib_cache[dataroot] = None
    return _calib_cache[dataroot]
