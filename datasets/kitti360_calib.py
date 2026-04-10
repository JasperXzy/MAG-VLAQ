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


def load_calibration(calib_dir):
    """Load KITTI-360 calibration from directory.

    Returns dict with:
      - T_velo_to_cam2: 4x4 transformation from velodyne to camera 02 frame
      - fisheye: dict of fisheye intrinsic parameters for cam 02
      - orig_width, orig_height: original fisheye image dimensions
    """
    calib_cam_to_velo_path = os.path.join(calib_dir, 'calib_cam_to_velo.txt')
    calib_cam_to_pose_path = os.path.join(calib_dir, 'calib_cam_to_pose.txt')
    fisheye_path = os.path.join(calib_dir, 'image_02.yaml')

    for p in [calib_cam_to_velo_path, calib_cam_to_pose_path, fisheye_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Calibration file not found: {p}\n"
                "Download KITTI-360 calibration from: "
                "https://www.cvlibs.net/datasets/kitti-360/download.php"
            )

    # calib_cam_to_velo: transforms cam0 points to velodyne frame (3x4)
    lastrow = np.array([[0, 0, 0, 1]])
    T_cam0_to_velo = np.concatenate(
        (np.loadtxt(calib_cam_to_velo_path).reshape(3, 4), lastrow)
    )

    # calib_cam_to_pose: transforms each camera to pose (vehicle) frame
    with open(calib_cam_to_pose_path, 'r') as fid:
        T_cam0_to_pose = np.concatenate((_read_variable(fid, 'image_00', 3, 4), lastrow))
        T_cam2_to_pose = np.concatenate((_read_variable(fid, 'image_02', 3, 4), lastrow))

    # Chain: velo -> cam0 -> pose -> cam2
    # p_cam0 = inv(T_cam0_to_velo) @ p_velo
    # p_pose = T_cam0_to_pose @ p_cam0
    # p_cam2 = inv(T_cam2_to_pose) @ p_pose
    T_velo_to_cam2 = np.linalg.inv(T_cam2_to_pose) @ T_cam0_to_pose @ np.linalg.inv(T_cam0_to_velo)

    # Fisheye intrinsics
    fi = _read_yaml_file(fisheye_path)

    calib = {
        'T_velo_to_cam2': T_velo_to_cam2,
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
    return calib


def project_velo_to_fisheye(points, calib, img_w, img_h):
    """Project velodyne 3D points to fisheye camera 02 pixel coordinates.

    Args:
        points: [N, 3] numpy array of velodyne XYZ coordinates
        calib: calibration dict from load_calibration()
        img_w: actual (resized) image width
        img_h: actual (resized) image height

    Returns:
        u, v: [N] pixel coordinates in the resized image
        valid: [N] boolean mask for points with valid projection
    """
    N = points.shape[0]

    # Transform to cam2 frame
    pts_hom = np.concatenate([points, np.ones((N, 1))], axis=1)  # [N, 4]
    pts_cam = (calib['T_velo_to_cam2'] @ pts_hom.T).T[:, :3]  # [N, 3]

    # Only keep points in front of camera (positive Z)
    depth = pts_cam[:, 2]
    valid = depth > 0.1

    # Fisheye (MEI omnidirectional) projection
    norm = np.linalg.norm(pts_cam, axis=1)  # [N]
    norm = np.clip(norm, 1e-8, None)

    x = pts_cam[:, 0] / norm
    y = pts_cam[:, 1] / norm
    z = pts_cam[:, 2] / norm

    xi = calib['xi']
    denom = z + xi
    denom = np.clip(np.abs(denom), 1e-8, None) * np.sign(denom + 1e-12)
    x = x / denom
    y = y / denom

    # Radial distortion
    k1, k2 = calib['k1'], calib['k2']
    ro2 = x * x + y * y
    distort = 1 + k1 * ro2 + k2 * ro2 * ro2
    x = x * distort
    y = y * distort

    # Apply intrinsics (in original image coordinates)
    u_orig = calib['gamma1'] * x + calib['u0']
    v_orig = calib['gamma2'] * y + calib['v0']

    # Scale to resized image
    scale_x = img_w / calib['orig_width']
    scale_y = img_h / calib['orig_height']
    u = u_orig * scale_x
    v = v_orig * scale_y

    # Update validity: must be within image bounds
    valid = valid & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)

    return u, v, valid


def colorize_points(points, image, calib):
    """Get per-point RGB colors by projecting 3D points onto a fisheye image.

    Args:
        points: [N, 3] numpy array of velodyne XYZ coordinates
        image: PIL Image or numpy array [H, W, 3] (uint8)
        calib: calibration dict from load_calibration()

    Returns:
        rgb: [N, 3] numpy array of RGB values normalized to [0, 1].
             Points outside the camera FOV have rgb = [0, 0, 0].
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))

    img_h, img_w = image.shape[:2]
    u, v, valid = project_velo_to_fisheye(points, calib, img_w, img_h)

    rgb = np.zeros((points.shape[0], 3), dtype=np.float32)
    if valid.any():
        u_int = np.clip(u[valid].astype(np.int32), 0, img_w - 1)
        v_int = np.clip(v[valid].astype(np.int32), 0, img_h - 1)
        rgb[valid] = image[v_int, u_int, :3].astype(np.float32) / 255.0

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
