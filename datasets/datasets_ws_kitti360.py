
import os
import torch
import faiss
import logging
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from os.path import join
import torch.utils.data as data
import torchvision.transforms as T
from torch.utils.data.dataset import Subset
from sklearn.neighbors import NearestNeighbors
from torch.utils.data.dataloader import DataLoader
import random
from torchvision.transforms import InterpolationMode
import torchvision.transforms as TVT
from viz_lidar import viz_lidar_open3d
import copy
import matplotlib.pyplot as plt
import utm
from pc_augmentation import (
    PCRandomFlip,
    PCRandomRotation,
    PCRandomTranslation,
    PCRandomScale,
    PCRandomShear,
    PCJitterPoints,
    PCRemoveRandomPoints,
    PCRemoveRandomBlock
)


from datasets.kitti360_calib import get_calibration, colorize_points
import open3d as o3d

from tools.options import parse_arguments
opt = parse_arguments()


# Utonia official input pipeline constants (transform.default(scale=0.2)
# + GridSample(grid_size=0.01)). Effective voxel = 0.01 / 0.2 = 0.05 m.
UTONIA_GLOBAL_SCALE = 0.2
UTONIA_GRID_SIZE = 0.01


def estimate_normals_o3d(pts, k=30):
    """Per-point normal estimation via Open3D, matching the Utonia paper:
        "Surface normals are estimated with Open3D, including directions
         from points to the LiDAR center."

    Args:
        pts: np.ndarray [N, 3] (velodyne XYZ)
        k: KNN neighborhood size (paper uses 30)
    Returns:
        normals: np.ndarray [N, 3] float32, oriented toward the LiDAR origin (0,0,0)
    """
    n = pts.shape[0]
    if n < 3:
        return np.zeros((n, 3), dtype=np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=min(k, n))
    )
    # Critical: orient every normal to point TOWARD the LiDAR sensor (origin in
    # velodyne frame). Without this, PCA-derived normals have random sign and
    # become a high-frequency noise channel for the pretrained Utonia encoder.
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))
    normals = np.asarray(pcd.normals, dtype=np.float32)
    return normals


def _rand_rotation_outdoor_scene():
    """Full yaw (z) rotation + mild roll/pitch perturbations, matching
    Utonia outdoor scene-level augmentation (Sec 4.1):
        z in [-pi, pi], x/y in [-pi/16, pi/16].
    """
    theta_z = 2.0 * np.pi * (np.random.rand() - 0.5)              # [-pi, pi]
    theta_x = (np.pi / 16.0) * 2.0 * (np.random.rand() - 0.5)     # [-pi/16, pi/16]
    theta_y = (np.pi / 16.0) * 2.0 * (np.random.rand() - 0.5)
    cz, sz = np.cos(theta_z), np.sin(theta_z)
    cx, sx = np.cos(theta_x), np.sin(theta_x)
    cy, sy = np.cos(theta_y), np.sin(theta_y)
    Rz = np.array([[cz, -sz, 0.0], [sz,  cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    Ry = np.array([[cy,  0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


def _rand_scale_jitter(eta=1.1):
    """Isotropic multiplicative rescale r ~ exp(U(-log eta, log eta)).
    Matches Sec A.2 / Sec 4.1: ±10% scale jitter for scene-level data.
    """
    log_eta = np.log(eta)
    return float(np.exp(np.random.uniform(-log_eta, log_eta)))


def _rand_anisotropic_jitter(eta=1.1):
    """Anisotropic per-axis jitter j ~ exp(U(-log eta, log eta)^3), matching
    DINOv3-style coordinate augmentation used by Utonia (Sec 3.3, Sec A.2).
    """
    log_eta = np.log(eta)
    return np.exp(np.random.uniform(-log_eta, log_eta, size=3)).astype(np.float32)


def _causal_modality_blind(rgb, normal, p_drop_rgb=0.3, p_drop_normal=0.3):
    """Per-sample (per-data) causal modality blinding (Sec 3.1).
    Randomly zero out the entire RGB or Normal modality for a sample, so the
    model does not over-rely on any single modality. Matches the
    with-color / without-color partitioning in Tab. 7(d).
    """
    if np.random.rand() < p_drop_rgb:
        rgb = np.zeros_like(rgb)
    if np.random.rand() < p_drop_normal:
        normal = np.zeros_like(normal)
    return rgb, normal


def _dedup_grid(coords_int_np):
    """Return indices to keep (order-preserving) so that int grid coords are unique."""
    _, keep = np.unique(coords_int_np, axis=0, return_index=True)
    keep.sort()
    return keep







trainselectlocationlist = [  # define which location to use
    "2013_05_28_drive_0000_sync",
    # "2013_05_28_drive_0002_sync",
    "2013_05_28_drive_0003_sync",
    "2013_05_28_drive_0004_sync",
    "2013_05_28_drive_0005_sync",
    "2013_05_28_drive_0006_sync",
    "2013_05_28_drive_0007_sync",
    # "2013_05_28_drive_0009_sync",
    "2013_05_28_drive_0010_sync",
]

testselectlocationlist = [
    "2013_05_28_drive_0000_sync",
    # "2013_05_28_drive_0002_sync",
    "2013_05_28_drive_0003_sync",
    "2013_05_28_drive_0004_sync",
    "2013_05_28_drive_0005_sync",
    "2013_05_28_drive_0006_sync",
    "2013_05_28_drive_0007_sync",
    # "2013_05_28_drive_0009_sync",
    "2013_05_28_drive_0010_sync",
]


train_ratio = opt.train_ratio
share_db = opt.share_db


base_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def path_to_pil_img(path):
    image = Image.open(path)
    image = image.convert("RGB")
    return image




def kitti360_collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (images,
        triplets_local_indexes, triplets_global_indexes).
        triplets_local_indexes are the indexes referring to each triplet within images.
        triplets_global_indexes are the global indexes of each image.
    Args:
        batch: list of tuple (images, triplets_local_indexes, triplets_global_indexes).
            considering each query to have 10 negatives (negs_num_per_query=10):
            - images: torch tensor of shape (12, 3, h, w).
            - triplets_local_indexes: torch tensor of shape (10, 3).
            - triplets_global_indexes: torch tensor of shape (12).
    Returns:
        images: torch tensor of shape (batch_size*12, 3, h, w).
        triplets_local_indexes: torch tensor of shape (batch_size*10, 3).
        triplets_global_indexes: torch tensor of shape (batch_size, 12).
    """
    # image
    # bev
    # pc
    # images = torch.cat([e[0]['image'] for e in batch]) # [12,c,h,w] -> [b*12,c,h,w]
    # bevs = torch.stack([e[0]['bev'] for e in batch]) # [3,h,w] -> [b,3,h,w]
    # sphs = torch.stack([e[0]['sph'] for e in batch]) # [3,h,w] -> [b,3,h,w]
    query_image = torch.stack([e[0]['query_image'] for e in batch]) 
    query_eastnorth = torch.stack([e[0]['query_eastnorth'] for e in batch]).float() # [b,2]
    db_map = torch.stack([e[0]['db_map'] for e in batch])
    db_eastnorth = torch.stack([e[0]['db_eastnorth'] for e in batch]).float() # [b,ndb,2]

    # ---- batch augmentation (quantize is in __getitem__)

    query_pc = [e[0]['query_pc'] for e in batch]
    query_bev = torch.stack([e[0]['query_bev'] for e in batch])
    query_sph = torch.stack([e[0]['query_sph'] for e in batch])

    # Per-point RGB and normals (normals precomputed in __getitem__)
    query_pc_rgb = [e[0]['query_pc_rgb'] for e in batch]
    query_pc_normal = [e[0]['query_pc_normal'] for e in batch]

    # Per-sample: apply SAME small rotation to coords and normals, then dedup int grid
    utonia_coord_list, utonia_grid_list, utonia_rgb_list, utonia_normal_list = [], [], [], []
    utonia_point_counts, utonia_batch_ids = [], []
    me_coord_list, me_batch_ids = [], []
    for i, p in enumerate(query_pc):
        p_np = p if isinstance(p, np.ndarray) else p.numpy()
        p_np = p_np.astype(np.float32, copy=False)
        r_np = query_pc_rgb[i]
        if not isinstance(r_np, np.ndarray):
            r_np = r_np.numpy()
        r_np = r_np.astype(np.float32, copy=False)
        n_np = query_pc_normal[i]
        if not isinstance(n_np, np.ndarray):
            n_np = n_np.numpy()
        n_np = n_np.astype(np.float32, copy=False)

        # Scene-level rigid augmentation (Utonia paper Sec 4.1):
        # full yaw (z) + mild roll/pitch, isotropic scale jitter (±10%),
        # plus DINOv3-style anisotropic per-axis jitter (Sec 3.3 / A.2).
        R = _rand_rotation_outdoor_scene()          # [3,3]
        s = _rand_scale_jitter(eta=1.1)             # scalar
        aniso = _rand_anisotropic_jitter(eta=1.1)   # [3]
        p_rot = (p_np @ R) * (s * aniso)
        # Normals are rotation-only (scale does not affect unit normals);
        # keep them unit-length after the rigid rotation.
        n_rot = n_np @ R

        # Per-sample Causal Modality Blinding (Sec 3.1): randomly zero RGB or normal
        r_np_b, n_rot_b = _causal_modality_blind(r_np, n_rot, p_drop_rgb=0.3, p_drop_normal=0.3)

        # Utonia official input: first global scale 0.2, then floor(coord / 0.01).
        # Effective voxel size = 0.05 m (real physical scale).
        p_scaled = p_rot * UTONIA_GLOBAL_SCALE
        p_int = np.floor(p_scaled / UTONIA_GRID_SIZE).astype(np.int32)

        keep = _dedup_grid(p_int)
        p_scaled_k = p_scaled[keep]
        p_int_k = p_int[keep]
        r_k = r_np_b[keep] if r_np_b.shape[0] == p_np.shape[0] else np.zeros((len(keep), 3), dtype=np.float32)
        n_k = n_rot_b[keep]

        n_pts = len(keep)
        utonia_coord_list.append(torch.from_numpy(p_scaled_k))
        utonia_grid_list.append(torch.from_numpy(p_int_k))
        utonia_rgb_list.append(torch.from_numpy(r_k))
        utonia_normal_list.append(torch.from_numpy(n_k))
        utonia_point_counts.append(n_pts)
        utonia_batch_ids.append(torch.full((n_pts, 1), i, dtype=torch.int32))

        # For ME (MinkFPN fallback): keep meter units + floor (not truncate)
        me_int = np.floor(p_rot).astype(np.int32)
        me_coord_list.append(torch.from_numpy(me_int))
        me_batch_ids.append(torch.full((me_int.shape[0], 1), i, dtype=torch.int32))

    utonia_coord = torch.cat(utonia_coord_list, dim=0).float()
    utonia_grid_coord = torch.cat(utonia_grid_list, dim=0).int()
    utonia_rgb = torch.cat(utonia_rgb_list, dim=0).float()
    utonia_normal = torch.cat(utonia_normal_list, dim=0).float()
    utonia_offset = torch.cumsum(torch.tensor(utonia_point_counts), dim=0)

    # ME inputs (kept for MinkFPN fallback)
    coords = torch.cat([torch.cat(me_batch_ids, dim=0), torch.cat(me_coord_list, dim=0)], dim=1)
    feats = torch.ones([coords.shape[0], 1]).float()
    utonia_feat = utonia_coord.clone()  # placeholder, overwritten inside UtoniaFE


    triplets_local_indexes = torch.cat([e[1][None] for e in batch])
    triplets_global_indexes = torch.cat([e[2][None] for e in batch])
    for i, (local_indexes, global_indexes) in enumerate(zip(triplets_local_indexes, triplets_global_indexes)):
        local_indexes += len(global_indexes) * i  # Increment local indexes by offset (len(global_indexes) is 12)
    output_dict = {
        # 'image': images,
        # 'bev': bevs,
        # 'sph': sphs,
        'coords': coords,
        'features': feats,
        'query_image': query_image,
        'query_bev': query_bev,
        'query_sph': query_sph,
        'query_eastnorth': query_eastnorth,
        # 'positive_db_map': positive_db_map,
        # 'negative_db_maps': negative_db_maps,
        'db_map': db_map,
        'db_eastnorth': db_eastnorth,
        'utonia_coord': utonia_coord,                 # [N_dedup, 3] float xyz (rotated)
        'utonia_grid_coord': utonia_grid_coord,       # [N_dedup, 3] int32 (deduped)
        'utonia_feat': utonia_feat,                   # placeholder, overwritten in UtoniaFE
        'utonia_rgb': utonia_rgb,                     # [N_dedup, 3] projected RGB (0-1)
        'utonia_normal': utonia_normal,               # [N_dedup, 3] precomputed normals
        'utonia_offset': utonia_offset,
    }
    # return images, torch.cat(tuple(triplets_local_indexes)), triplets_global_indexes
    return output_dict, torch.cat(tuple(triplets_local_indexes)), triplets_global_indexes





def kitti360_collate_fn_cache_db(batch):
    # images = torch.stack([e[0]['image'] for e in batch]) # [12,c,h,w] -> [b*12,c,h,w]
    # bevs = torch.empty([images.shape[0], 0])
    # sphs = torch.empty([images.shape[0], 0])
    # coords = torch.empty([images.shape[0], 0])
    # feats = torch.empty([images.shape[0], 0])
    # query_image = torch.stack([e[0]['query_image'] for e in batch])
    # positive_db_map = torch.stack([e[0]['positive_db_map'] for e in batch])
    # negative_db_maps = torch.stack([e[0]['negative_db_maps'] for e in batch])
    # query_image = torch.stack([e[0]['query_image'] for e in batch])
    # query_bev = torch.stack([e[0]['query_bev'] for e in batch])
    db_map = torch.stack([e[0]['db_map'] for e in batch])
    indices = torch.tensor([e[1] for e in batch])

    output_dict = {
        # 'image': images,
        # 'bev': bevs,
        # 'sph': sphs,
        # 'coord': coords,
        # 'feat': feats,
        # 'query_image': query_image,
        # 'positive_db_map': positive_db_map,
        # 'negative_db_maps': negative_db_maps,
        # 'query_image': query_image,
        # 'query_bev': query_bev,
        'db_map': db_map,
    }
    return output_dict, indices




def kitti360_collate_fn_cache_q(batch):
    '''
    output of collate_fn should be applicable with .to(device)
    '''
    # images = torch.stack([e[0]['image'] for e in batch]) # [12,c,h,w] -> [b*12,c,h,w]
    # bevs = torch.stack([e[0]['bev'] for e in batch]) # [3,h,w] -> [b,3,h,w]
    # # sphs = torch.stack([e[0]['sph'] for e in batch]) # [3,h,w] -> [b,3,h,w]
    # pcs = [e[0]['pc'] for e in batch]
    query_eastnorth = torch.stack([e[0]['query_eastnorth'] for e in batch])
    query_image = torch.stack([e[0]['query_image'] for e in batch])
    # ---- batch augmentation (quantize is in __getitem__)

    query_pc = [e[0]['query_pc'] for e in batch]
    query_bev = torch.stack([e[0]['query_bev'] for e in batch])
    query_sph = torch.stack([e[0]['query_sph'] for e in batch])

    query_pc_rgb = [e[0]['query_pc_rgb'] for e in batch]
    query_pc_normal = [e[0]['query_pc_normal'] for e in batch]

    # Per-sample dedup (no rotation at inference); precomputed normals from __getitem__
    utonia_coord_list, utonia_grid_list, utonia_rgb_list, utonia_normal_list = [], [], [], []
    utonia_point_counts = []
    me_coord_list, me_batch_ids = [], []
    for i, p in enumerate(query_pc):
        p_np = p if isinstance(p, np.ndarray) else p.numpy()
        p_np = p_np.astype(np.float32, copy=False)
        r_np = query_pc_rgb[i]
        if not isinstance(r_np, np.ndarray):
            r_np = r_np.numpy()
        r_np = r_np.astype(np.float32, copy=False)
        n_np = query_pc_normal[i]
        if not isinstance(n_np, np.ndarray):
            n_np = n_np.numpy()
        n_np = n_np.astype(np.float32, copy=False)

        # Utonia official input (no augmentation): global scale 0.2 then floor / 0.01.
        p_scaled = p_np * UTONIA_GLOBAL_SCALE
        p_int = np.floor(p_scaled / UTONIA_GRID_SIZE).astype(np.int32)

        keep = _dedup_grid(p_int)
        p_k = p_scaled[keep]
        g_k = p_int[keep]
        r_k = r_np[keep] if r_np.shape[0] == p_np.shape[0] else np.zeros((len(keep), 3), dtype=np.float32)
        n_k = n_np[keep]

        n_pts = len(keep)
        utonia_coord_list.append(torch.from_numpy(p_k))
        utonia_grid_list.append(torch.from_numpy(g_k))
        utonia_rgb_list.append(torch.from_numpy(r_k))
        utonia_normal_list.append(torch.from_numpy(n_k))
        utonia_point_counts.append(n_pts)

        me_int = np.floor(p_np).astype(np.int32)
        me_coord_list.append(torch.from_numpy(me_int))
        me_batch_ids.append(torch.full((me_int.shape[0], 1), i, dtype=torch.int32))

    utonia_coord = torch.cat(utonia_coord_list, dim=0).float()
    utonia_grid_coord = torch.cat(utonia_grid_list, dim=0).int()
    utonia_rgb = torch.cat(utonia_rgb_list, dim=0).float()
    utonia_normal = torch.cat(utonia_normal_list, dim=0).float()
    utonia_offset = torch.cumsum(torch.tensor(utonia_point_counts), dim=0)

    coords = torch.cat([torch.cat(me_batch_ids, dim=0), torch.cat(me_coord_list, dim=0)], dim=1)
    feats = torch.ones([coords.shape[0], 1]).float()
    utonia_feat = utonia_coord.clone()


    db_map = torch.stack([e[0]['db_map'] for e in batch])
    # positive_db_map = torch.stack([e[0]['positive_db_map'] for e in batch])
    # negative_db_maps = torch.stack([e[0]['negative_db_maps'] for e in batch])

    indices = torch.tensor([e[1] for e in batch])
    output_dict = {
        # 'image': images,
        # 'bev': bevs,
        # 'sph': sphs,
        'coords': coords,
        'features': feats,
        'query_image': query_image,
        'query_bev': query_bev,
        'query_sph': query_sph,
        # 'positive_db_map': positive_db_map,
        # 'negative_db_maps': negative_db_maps,
        'db_map': db_map,
        'query_eastnorth': query_eastnorth,
        'utonia_coord': utonia_coord,
        'utonia_grid_coord': utonia_grid_coord,
        'utonia_feat': utonia_feat,
        'utonia_rgb': utonia_rgb,
        'utonia_normal': utonia_normal,
        'utonia_offset': utonia_offset,
    }
    return output_dict, indices







def load_qimage(datapath, split):
    image = Image.open(datapath)
    image = image.convert('RGB')
    if split == 'train':
        tf = TVT.Compose([TVT.Resize((opt.q_resize, opt.q_resize)),
                        TVT.ColorJitter(brightness=opt.q_jitter, contrast=opt.q_jitter, saturation=opt.q_jitter, hue=min(0.5, opt.q_jitter)),
                        TVT.ToTensor(),
                        #   TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        TVT.Normalize(mean=0.5, std=0.22)
                        ])
    elif split == 'test':
        tf = TVT.Compose([TVT.Resize((opt.q_resize, opt.q_resize)),
                        TVT.ToTensor(),
                        #   TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        TVT.Normalize(mean=0.5, std=0.22)
                        ])
    image = tf(image)
    return image



def load_dbimage(datapath, split):
    image = Image.open(datapath)
    image = image.convert('RGB')
    if split == 'train':
        tf = TVT.Compose([
            TVT.CenterCrop(opt.db_cropsize),
            TVT.Resize((opt.db_resize, opt.db_resize)),
            TVT.ColorJitter(brightness=opt.db_jitter, contrast=opt.db_jitter, saturation=opt.db_jitter, hue=min(0.5, opt.db_jitter)),
            TVT.ToTensor(),
        #   TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            TVT.Normalize(mean=0.5, std=0.22)
                        ])
    elif split == 'test':
        tf = TVT.Compose([
            TVT.CenterCrop(opt.db_cropsize),
            TVT.Resize((opt.db_resize, opt.db_resize)),
            TVT.ToTensor(),
        #   TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            TVT.Normalize(mean=0.5, std=0.22)
                        ])
    else:
        raise NotImplementedError
    image = tf(image)
    return image





def generate_bev_from_pc(pc, w=200, max_thd=100):
    """
    w+1: bev width
    max_thd: max threshold of x,y,z
    """
    pc = copy.deepcopy(pc)
    assert pc.shape[1] == 3
    # remove pc outside max_thd
    pc = pc[np.max(np.abs(pc), axis=1) < max_thd]
    bin_max = np.max(pc, axis=0)
    bin_min = np.min(pc, axis=0)
    assert np.all(bin_max <= max_thd)
    assert np.all(bin_min >= -max_thd)
    pc = pc + max_thd
    pc = pc / (2 * max_thd) * w
    pc = pc.astype(np.int64)
    bin_max = np.max(pc, axis=0)
    # print(bin_max)
    bev = np.zeros([w+1, w+1], dtype=np.float32)
    assert np.all(bin_max <= bev.shape[0])
    bev[pc[:, 0], pc[:, 1]] = pc[:, 2]
    return bev




def generate_sph_from_pc(pc, w=361, h=61):
    # kitti   w=361  h=61
    # ithaca  w=361  h=101
    # kitti360 w=361 h=61
    # w = 361
    # h = 61
    # if 'ithaca365' in opt.dataset_name:
    #     w = 361
    #     h = 101

    # generate spherical projection from pc
    pc = copy.deepcopy(pc)
    assert pc.shape[1] == 3
    # u-v : h-w
    # u: 
    u = np.arctan2(pc[:,2], np.sqrt(pc[:,0]**2 + pc[:,1]**2))
    u = u / np.pi * 180
    u = u + 25 
    u = u * 2
    u = h - u
    # v: [0, 360]
    v = np.arctan2(pc[:,0], pc[:,1])
    v = v / np.pi * 180
    v = v + 180
    r = np.sqrt(pc[:,0]**2 + pc[:,1]**2 + pc[:,2]**2)
    uv = np.stack([u, v], axis=1)
    uv = np.array(uv, dtype=np.int32)
    # plt.scatter(uv[:,1], uv[:,0], s=1, c=r, cmap='jet')
    # plt.show()
    if 'ithaca365' in opt.dataset_name:
        ids_h = (uv[:,0] < h) &  (uv[:,0] >= 0)
        uv = uv[ids_h]
        r = r[ids_h]
    sph = np.zeros([h, w])
    sph[uv[:,0], uv[:,1]] = r
    # if 'ithaca365' in opt.dataset_name:
    #     sph = sph[25:80]
    # plt.imshow(sph)
    # plt.imsave('sph.png', sph)
    # plt.close()
    # plt.show()
    return sph
    




def load_pc_sph_bev(file_path, split): # filename is the same as load_pc
    pc = np.fromfile(file_path, dtype=np.float32).reshape(-1,3) # for kitti360_voxel
    # ==== sph
    if 'lcpr' in opt.modelq or 'liploc' in opt.modelq:
        sph = generate_sph_from_pc(pc, w=361, h=61)
        sph = Image.fromarray(sph).convert('RGB')
        # (_w,_h) = sph.size
        # assert opt.sph_resize <= 1
        # resize_ratio = random.uniform(opt.sph_resize, 2-opt.sph_resize)
        # resize_size = int(resize_ratio*min(_w,_h))
        if split == 'train':
            tf = TVT.Compose([
                TVT.Resize(opt.sph_size, interpolation=InterpolationMode.NEAREST),
                TVT.ColorJitter(brightness=opt.sph_jit, contrast=opt.sph_jit, saturation=opt.sph_jit, hue=min(0.5, opt.sph_jit)),
                TVT.ToTensor(),
                # TVT.Normalize(mean=opt.sph_mean, std=opt.sph_std)
            ])
        elif split == 'test':
            tf = TVT.Compose([
                TVT.Resize(opt.sph_size, interpolation=InterpolationMode.NEAREST),
                # TVT.Resize(resize_size, interpolation=InterpolationMode.NEAREST),
                TVT.ToTensor(),
                # TVT.Normalize(mean=opt.sph_mean, std=opt.sph_std)
            ])
        else:
            raise NotImplementedError
        sph = tf(sph)
    else:
        sph = torch.empty(0)
    # ==== bev
    if 'bevplace' in opt.modelq:
        bev = generate_bev_from_pc(pc, w=200, max_thd=100)
        bev = Image.fromarray(bev).convert('RGB')
        if split == 'train':
            tf = TVT.Compose([
                # TVT.CenterCrop(opt.db_cropsize),
                # TVT.Resize(opt.db_resize), 
                TVT.ColorJitter(brightness=opt.bev_jit, contrast=opt.bev_jit, saturation=opt.bev_jit, hue=min(0.5, opt.bev_jit)),
                TVT.ToTensor(),
            ])
        elif split == 'test':
            tf = TVT.Compose([
                # TVT.CenterCrop(opt.db_cropsize),
                # TVT.Resize(opt.db_resize), 
                TVT.ToTensor(),
            ])
        else:
            raise NotImplementedError
            
        bev = tf(bev)
    else:
        bev = torch.empty(0)

    return pc, sph, bev




def load_pc_sph(file_path, split): # filename is the same as load_pc
    pc = np.fromfile(file_path, dtype=np.float32).reshape(-1,3) # for kitti360_voxel
    sph = generate_sph_from_pc(pc, w=361, h=61)
    # ==== sph
    sph = Image.fromarray(sph).convert('RGB')
    # (_w,_h) = sph.size
    # assert opt.sph_resize <= 1
    # resize_ratio = random.uniform(opt.sph_resize, 2-opt.sph_resize)
    # resize_size = int(resize_ratio*min(_w,_h))
    if split == 'train':
        tf = TVT.Compose([
            # TVT.Resize(resize_size, interpolation=InterpolationMode.NEAREST),
            TVT.ColorJitter(brightness=opt.sph_jit, contrast=opt.sph_jit, saturation=opt.sph_jit, hue=min(0.5, opt.sph_jit)),
            TVT.ToTensor(),
            # TVT.Normalize(mean=opt.sph_mean, std=opt.sph_std)
        ])
    elif split == 'test':
        tf = TVT.Compose([
            # TVT.Resize(resize_size, interpolation=InterpolationMode.NEAREST),
            TVT.ToTensor(),
            # TVT.Normalize(mean=opt.sph_mean, std=opt.sph_std)
        ])
    else:
        raise NotImplementedError
    sph = tf(sph)
    return pc, sph







class PCADataset(data.Dataset):
    def __init__(self, args, datasets_folder="dataset", dataset_folder="pitts30k/images/train"):
        dataset_folder_full_path = join(datasets_folder, dataset_folder)
        if not os.path.exists(dataset_folder_full_path):
            raise FileNotFoundError(f"Folder {dataset_folder_full_path} does not exist")
        self.images_paths = sorted(glob(join(dataset_folder_full_path, "**", "*.jpg"), recursive=True))
    
    def __getitem__(self, index):
        data_dict = {
            'image': base_transform(path_to_pil_img(self.images_paths[index]))
        }
        # return base_transform(path_to_pil_img(self.images_paths[index]))
        return data_dict
    
    def __len__(self):
        return len(self.images_paths)









# =================== test/cache dataset

class KITTI360BaseDataset(data.Dataset):
    """Dataset with images from database and queries, used for inference (testing and building cache).
    """
    def __init__(self, args, datasets_folder="datasets", dataset_name="pitts30k", split="train"):
        super().__init__()
        self.dataset_name = dataset_name
        
        self.resize = args.resize
        self.test_method = args.test_method
        self.split = split
        

        # ==========
        # Use  v1.0-trainval_singapore-onenorth  training
        dataroot = opt.dataroot
        if split == 'train':
            selectlocationlist = trainselectlocationlist
        elif split == 'test':
            selectlocationlist = testselectlocationlist
        else:
            raise NotImplementedError
        self.queries_infos = []
        self.queries_utms = []
        resize = 320
        for selectlocation in selectlocationlist:
            print(selectlocation)
            qpcdir = os.path.join(dataroot, 'data_3d_voxel0.5', selectlocation, 'velodyne_points/data')
            qposedir = os.path.join(dataroot, 'data_poses', selectlocation, 'oxts/data')
            qimage00dir = os.path.join(dataroot, f'data_2d_raw_resize{resize}', selectlocation, 'image_00/data_rect')
            qimage02dir = os.path.join(dataroot, f'data_2d_raw_resize{resize}', selectlocation, 'image_02/data_rgb')
            qimage03dir = os.path.join(dataroot, f'data_2d_raw_resize{resize}', selectlocation, 'image_03/data_rgb')
            qimage0203dir = os.path.join(dataroot, 'data_2d_cat0203', selectlocation, 'image_0203/data_rgb')
            qpcnames = sorted(os.listdir(qpcdir))
            qimage00names = sorted(os.listdir(qimage00dir))
            qimage0203names = sorted(os.listdir(qimage0203dir))
            assert len(qpcnames) == len(qimage00names)
            assert len(qpcnames) == len(qimage0203names)
            if split == 'train':
                qimage0203names = qimage0203names[:int(len(qimage0203names)*train_ratio)]
            elif split == 'test':
                qimage0203names = qimage0203names[int(len(qimage0203names)*train_ratio):]
            print(f"Number of q samples in {selectlocation}: {len(qimage0203names)}")
            for i_sample, qimage0203name in enumerate(qimage0203names):
                if split == 'train':
                    if i_sample % opt.traindownsample!= 0: # using 1/2 samples for training
                        continue
                elif split == 'test': # using all samples for testing
                    None
                else:
                    raise NotImplementedError
                qpcpath = os.path.join(qpcdir, qimage0203name.replace('.png','.bin'))
                qimage00path = os.path.join(qimage00dir, qimage0203name.replace('.png','.png'))
                qimage02path = os.path.join(qimage02dir, qimage0203name.replace('.png','.png'))
                qimage03path = os.path.join(qimage03dir, qimage0203name.replace('.png','.png'))
                qimage0203path = os.path.join(qimage0203dir, qimage0203name.replace('.png','.png'))
                qposepath = os.path.join(qposedir, qimage0203name.replace('.png','.txt'))
                # if not os.path.exists(qpcpath): continue
                # if not os.path.exists(qposepath): continue
                qpose = open(qposepath).readline().split(' ')
                lat, lon = float(qpose[0]), float(qpose[1])
                east, north, _, _ = utm.from_latlon(lat, lon)
                qsampleinfo = {
                    'lat': lat,
                    'lon': lon,
                    'east': east,
                    'north': north,
                    'qposepath': qposepath,
                    'qimage00path': qimage00path,
                    'qimage02path': qimage02path,
                    'qimage03path': qimage03path,
                    'qimage0203path': qimage0203path,
                    'qpcpath': qpcpath,
                    'location': selectlocation,
                }
                self.queries_infos.append(qsampleinfo)
                self.queries_utms.append([east, north])
        self.queries_utms = np.array(self.queries_utms, dtype=np.float32)
        print(f"Number of q samples in {split}: {len(self.queries_infos)}")



        # Merge all db
        self.database_utms = []
        self.database_infos = []
        scale = 1
        zoom = 20   # higher is closer
        size = 320
        # maptype = opt.maptype
        # maptype = 'satellite'
        # maptype = 'osm'
        for selectlocation in selectlocationlist:
            # if maptype in ['satellite','roadmap']:
            #     dbdir = os.path.join(dataroot, f'data_aerial_{scale}_{zoom}_{size}_{maptype}', selectlocation)
            # elif maptype in ['osm']:
            #     dbdir = os.path.join(dataroot, f'data_aerial_19_{size}_{maptype}', selectlocation)
            db_satellite_dir = os.path.join(dataroot, f'data_aerial_{scale}_{zoom}_{size}_satellite', selectlocation)
            db_roadmap_dir = os.path.join(dataroot, f'data_aerial_{scale}_{zoom}_{size}_roadmap', selectlocation)
            # db_osm_dir = os.path.join(dataroot, f'data_aerial_19_{size}_osm', selectlocation)
            dbnames = os.listdir(db_satellite_dir)
            dbnames = sorted(dbnames)
            if share_db == True:
                dbnames = dbnames
            elif share_db == False:
                if split == 'train':
                    dbnames = dbnames[:int(len(dbnames)*train_ratio)]
                elif split == 'test':
                    dbnames = dbnames[int(len(dbnames)*train_ratio):]
            for i_dbname, dbname in enumerate(dbnames):
                if split == 'train':
                    if i_dbname % opt.traindownsample != 0: 
                        continue
                elif split == 'test':
                    None
                dbname_pure = dbname.replace('.png','')
                dbeastnorth = dbname_pure.split('@')[1:3]
                dblatlon = dbname_pure.split('@')[3:5]
                east, north = float(dbeastnorth[0]), float(dbeastnorth[1])
                lat, lon = float(dblatlon[0]), float(dblatlon[1])
                db_satellite_path = os.path.join(db_satellite_dir, dbname)
                db_roadmap_path = os.path.join(db_roadmap_dir, dbname)
                dbsampleinfo = {
                    'lat': lat,
                    'lon': lon,
                    'east': east,
                    'north': north,
                    'db_satellite_path': db_satellite_path,
                    'db_roadmap_path': db_roadmap_path,
                    'location': selectlocation,
                }
                self.database_infos.append(dbsampleinfo)
                self.database_utms.append([east, north])
        self.database_utms = np.array(self.database_utms, dtype=np.float32)
        print(f"Number of aerial db in {split}: {len(self.database_infos)}")


        # Find positive and negative 
        knn = NearestNeighbors(n_jobs=opt.num_workers+1)
        knn.fit(self.database_utms)
        softposthd = opt.val_positive_dist_threshold  # 25
        self.soft_positives_per_query = knn.radius_neighbors(self.queries_utms,
                                                             radius=softposthd,
                                                             return_distance=False)
        
        self.database_queries_infos = self.database_infos + self.queries_infos  # db + q
        self.database_num = len(self.database_infos)
        self.queries_num = len(self.queries_infos)
        a=1

    
    def __getitem__(self, index):
        query_pc_rgb = np.zeros((1, 3), dtype=np.float32)  # default
        query_pc_normal = np.zeros((1, 3), dtype=np.float32)
        if index >= self.database_num: # query
            # print(index)
            if opt.camnames == '00':
                query_image = load_qimage(datapath=self.queries_infos[index-self.database_num]['qimage00path'],split=self.split)
            elif opt.camnames == '0203':
                query_image = load_qimage(datapath=self.database_queries_infos[index]['qimage0203path'],split=self.split)
            else:
                raise NotImplementedError
            if opt.read_pc == True:
                query_sph, query_bev = torch.empty(0), torch.empty(0)
                query_pc, query_sph, query_bev = load_pc_sph_bev(file_path=self.database_queries_infos[index]['qpcpath'],split=self.split)
                # Project RGB from fisheye camera to point cloud
                calib = get_calibration(opt.dataroot)
                if calib is not None:
                    info = self.database_queries_infos[index]
                    img02 = Image.open(info['qimage02path']).convert('RGB')
                    img03 = Image.open(info['qimage03path']).convert('RGB')
                    query_pc_rgb = colorize_points(query_pc, img02, img03, calib)
                else:
                    query_pc_rgb = np.zeros((query_pc.shape[0], 3), dtype=np.float32)
                # Precompute per-point normals once (CPU, parallel via num_workers)
                query_pc_normal = estimate_normals_o3d(query_pc, k=30)
            else:
                query_pc = torch.ones([1,3]).float()
                query_sph = torch.empty(0)
                query_bev = torch.empty(0)
            # query_bev = torch.empty(0)
            # query_pc = load_pc(file_path=self.database_queries_infos[index]['qpcpath'],split=self.split)
            # db_satellite_map = torch.empty(0)
            # db_roadmap_map = torch.empty(0)
            # db_map = torch.stack([db_satellite_map, db_roadmap_map], 0) # [nmap,3,h,w]
            db_map = torch.empty(0)
            query_eastnorth = torch.tensor([self.database_queries_infos[index]['east'], self.database_queries_infos[index]['north']])
        else: # database
            query_image = torch.empty(0)
            query_pc, query_bev, query_sph = torch.empty(0), torch.empty(0), torch.empty(0)
            maptype = opt.maptype.split('_')
            db_map = []
            for each_maptype in maptype:
                if each_maptype == 'satellite':
                    each_db_map = load_dbimage(datapath=self.database_queries_infos[index]['db_satellite_path'], split=self.split)
                elif each_maptype == 'roadmap':
                    each_db_map = load_dbimage(datapath=self.database_queries_infos[index]['db_roadmap_path'], split=self.split)
                db_map.append(each_db_map)
            db_map = torch.stack(db_map, 0) # [nmap,3,h,w]
            query_eastnorth = torch.empty(0)

        output_dict = {
            'query_image': query_image,
            'query_bev': query_bev,
            'query_sph': query_sph,
            'query_pc': query_pc,
            'query_pc_rgb': query_pc_rgb,
            'query_pc_normal': query_pc_normal,
            'db_map': db_map,
            'query_eastnorth': query_eastnorth
        }
        return output_dict, index


    
    def _test_query_transform(self, img):
        """Transform query image according to self.test_method."""
        C, H, W = img.shape
        if self.test_method == "single_query":
            # self.test_method=="single_query" is used when queries have varying sizes, and can't be stacked in a batch.
            processed_img = T.functional.resize(img, min(self.resize), antialias=True)
        elif self.test_method == "central_crop":
            # Take the biggest central crop of size self.resize. Preserves ratio.
            scale = max(self.resize[0]/H, self.resize[1]/W)
            processed_img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale).squeeze(0)
            processed_img = T.functional.center_crop(processed_img, self.resize)
            assert processed_img.shape[1:] == torch.Size(self.resize), f"{processed_img.shape[1:]} {self.resize}"
        elif self.test_method == "five_crops" or self.test_method == 'nearest_crop' or self.test_method == 'maj_voting':
            # Get 5 square crops with size==shorter_side (usually 480). Preserves ratio and allows batches.
            shorter_side = min(self.resize)
            processed_img = T.functional.resize(img, shorter_side)
            processed_img = torch.stack(T.functional.five_crop(processed_img, shorter_side))
            assert processed_img.shape == torch.Size([5, 3, shorter_side, shorter_side]), \
                f"{processed_img.shape} {torch.Size([5, 3, shorter_side, shorter_side])}"
        return processed_img
    
    def __len__(self):
        return len(self.database_queries_infos)
    
    def __repr__(self):
        return f"< {self.__class__.__name__}, {self.dataset_name} - #database: {self.database_num}; #queries: {self.queries_num} >"
    
    def get_positives(self):
        return self.soft_positives_per_query















# =================== train dataset

class KITTI360TripletsDataset(KITTI360BaseDataset):
    """Dataset used for training, it is used to compute the triplets
    with TripletsDataset.compute_triplets() with various mining methods.
    If is_inference == True, uses methods of the parent class BaseDataset,
    this is used for example when computing the cache, because we compute features
    of each image, not triplets.
    """
    def __init__(self, args, datasets_folder="datasets", dataset_name="pitts30k", split="train", negs_num_per_query=10):
        super().__init__(args, datasets_folder, dataset_name, split)
        self.mining = args.mining
        self.neg_samples_num = args.neg_samples_num  # Number of negatives to randomly sample
        self.negs_num_per_query = negs_num_per_query  # Number of negatives per query in each batch
        if self.mining == "full":  # "Full database mining" keeps a cache with last used negatives
            self.neg_cache = [np.empty((0,), dtype=np.int32) for _ in range(self.queries_num)]
        self.is_inference = False
        self.split = split


        # Find hard_positives_per_query, which are within train_positives_dist_threshold (10 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        hardposthd = opt.train_positives_dist_threshold
        self.hard_positives_per_query = list(knn.radius_neighbors(self.queries_utms,
                                             radius=hardposthd,  # 10 meters
                                             return_distance=False))
        
        #### Some queries might have no positive, we should remove those queries.
        queries_without_any_hard_positive = np.where(np.array([len(p) for p in self.hard_positives_per_query], dtype=object) == 0)[0]
        if len(queries_without_any_hard_positive) != 0:
            logging.info(f"There are {len(queries_without_any_hard_positive)} queries without any positives " +
                         "within the training set. They won't be considered as they're useless for training.")
        # Remove queries without positives
        # self.hard_positives_per_query = np.delete(self.hard_positives_per_query, queries_without_any_hard_positive)
        self.hard_positives_per_query = [p for i, p in enumerate(self.hard_positives_per_query) if i not in queries_without_any_hard_positive]


        self.queries_infos = [e for i, e in enumerate(self.queries_infos) if i not in queries_without_any_hard_positive]
        self.database_queries_infos = self.database_infos + self.queries_infos  # db + q
        self.queries_num = len(self.queries_infos)


        # msls_weighted refers to the mining presented in MSLS paper's supplementary.
        # Basically, images from uncommon domains are sampled more often. Works only with MSLS dataset.
        if self.mining == "msls_weighted":
            notes = [p.split("@")[-2] for p in self.queries_paths]
            try:
                night_indexes = np.where(np.array([n.split("_")[0] == "night" for n in notes]))[0]
                sideways_indexes = np.where(np.array([n.split("_")[1] == "sideways" for n in notes]))[0]
            except IndexError:
                raise RuntimeError("You're using msls_weighted mining but this dataset " +
                                   "does not have night/sideways information. Are you using Mapillary SLS?")
            self.weights = np.ones(self.queries_num)
            assert len(night_indexes) != 0 and len(sideways_indexes) != 0, \
                "There should be night and sideways images for msls_weighted mining, but there are none. Are you using Mapillary SLS?"
            self.weights[night_indexes] += self.queries_num / len(night_indexes)
            self.weights[sideways_indexes] += self.queries_num / len(sideways_indexes)
            self.weights /= self.weights.sum()
            logging.info(f"#sideways_indexes [{len(sideways_indexes)}/{self.queries_num}]; " +
                         "#night_indexes; [{len(night_indexes)}/{self.queries_num}]")
    

    def __getitem__(self, index):
        if self.is_inference:
            # At inference time return the single image. This is used for caching or computing NetVLAD's clusters
            return super().__getitem__(index)

        query_index, best_positive_index, neg_indexes = torch.split(self.triplets_global_indexes[index], (1, 1, self.negs_num_per_query))
        if opt.camnames == '00':
            query_image = load_qimage(datapath=self.queries_infos[query_index]['qimage00path'],split=self.split)
        elif opt.camnames == '0203':
            query_image = load_qimage(datapath=self.queries_infos[query_index]['qimage0203path'],split=self.split) # [3,h,w]
        else:
            raise NotImplementedError
        query_info = self.queries_infos[query_index]
        query_eastnorth = torch.tensor([query_info['east'], query_info['north']])

        if opt.read_pc == True:
            query_sph, query_bev = torch.empty(0), torch.empty(0)
            query_pc, query_sph, query_bev = load_pc_sph_bev(file_path=self.queries_infos[query_index]['qpcpath'],split=self.split)
            # Project RGB from fisheye camera to point cloud
            calib = get_calibration(opt.dataroot)
            if calib is not None:
                img02 = Image.open(self.queries_infos[query_index]['qimage02path']).convert('RGB')
                img03 = Image.open(self.queries_infos[query_index]['qimage03path']).convert('RGB')
                query_pc_rgb = colorize_points(query_pc, img02, img03, calib)
            else:
                query_pc_rgb = np.zeros((query_pc.shape[0], 3), dtype=np.float32)
            # Precompute per-point normals (CPU, parallelized by num_workers)
            query_pc_normal = estimate_normals_o3d(query_pc, k=30)
        # query_pc = load_pc(file_path=self.queries_infos[query_index]['qpcpath'],split=self.split)
        else:
            query_pc = torch.ones([1,3]).float()
            query_pc_rgb = np.zeros((1, 3), dtype=np.float32)
            query_pc_normal = np.zeros((1, 3), dtype=np.float32)
            query_sph = torch.empty(0)
            query_bev = torch.empty(0)


        # positive_db_map = load_dbimage(datapath=self.database_infos[best_positive_index]['dbpath'])  # [3,h,w]
        # positive_db_map = []
        # negative_db_map = []
        # if opt.maptype == 'satellite':
        #     positive_db_satellite_map = load_dbimage(datapath=self.database_infos[best_positive_index]['db_satellite_path'],split=self.split)  # [3,h,w]
        #     negative_db_satellite_map = [load_dbimage(datapath=self.database_infos[e]['db_satellite_path'],split=self.split) for e in neg_indexes] # [nneg,3,h,w]
        #     negative_db_satellite_map = torch.stack(negative_db_satellite_map, 0)  # [nneg,3,h,w]
        #     positive_db_map.append(positive_db_satellite_map)
        #     negative_db_map.append(negative_db_satellite_map)
        # elif opt.maptype == 'roadmap':
        #     positive_db_roadmap_map = load_dbimage(datapath=self.database_infos[best_positive_index]['db_roadmap_path'],split=self.split)
        #     negative_db_roadmap_map = [load_dbimage(datapath=self.database_infos[e]['db_roadmap_path'],split=self.split) for e in neg_indexes]
        #     negative_db_roadmap_map = torch.stack(negative_db_roadmap_map, 0)  # [nneg,3,h,w]
        #     positive_db_map.append(positive_db_roadmap_map)
        #     negative_db_map.append(negative_db_roadmap_map)

        # positive_db_satellite_map = load_dbimage(datapath=self.database_infos[best_positive_index]['db_satellite_path'],split=self.split)  # [3,h,w]
        # positive_db_roadmap_map = load_dbimage(datapath=self.database_infos[best_positive_index]['db_roadmap_path'],split=self.split)  # [3,h,w]
        # positive_db_map = torch.stack([positive_db_satellite_map, positive_db_roadmap_map], 0)  # [nmap,3,h,w]
        # # negative_db_map = [load_dbimage(datapath=self.database_infos[e]['dbpath']) for e in neg_indexes] # [nneg,3,h,w]
        # negative_db_satellite_map = [load_dbimage(datapath=self.database_infos[e]['db_satellite_path'],split=self.split) for e in neg_indexes] # [nneg,3,h,w]
        # negative_db_satellite_map = torch.stack(negative_db_satellite_map, 0)  # [nneg,3,h,w]
        # negative_db_roadmap_map = [load_dbimage(datapath=self.database_infos[e]['db_roadmap_path'],split=self.split) for e in neg_indexes] # [nneg,3,h,w]
        # negative_db_roadmap_map = torch.stack(negative_db_roadmap_map, 0)  # [nneg,3,h,w]
        # negative_db_map = torch.stack([negative_db_satellite_map, negative_db_roadmap_map], 1)  # [nneg,nmap,3,h,w]

        maptype = opt.maptype.split('_')
        positive_db_map = []
        negative_db_map = []
        for each_maptype in maptype:
            if each_maptype == 'satellite':
                each_positive_db_map = load_dbimage(datapath=self.database_infos[best_positive_index]['db_satellite_path'],split=self.split)
                each_negative_db_map = [load_dbimage(datapath=self.database_infos[e]['db_satellite_path'],split=self.split) for e in neg_indexes]
                # each_positive_db_info = self.database_infos[best_positive_index]
                # each_negative_db_info = [self.database_infos[e] for e in neg_indexes]
            elif each_maptype == 'roadmap':
                each_positive_db_map = load_dbimage(datapath=self.database_infos[best_positive_index]['db_roadmap_path'],split=self.split)
                each_negative_db_map = [load_dbimage(datapath=self.database_infos[e]['db_roadmap_path'],split=self.split) for e in neg_indexes]
                # each_positive_db_info = self.database_infos[best_positive_index]
                # each_negative_db_info = [self.database_infos[e] for e in neg_indexes]

            each_negative_db_map = torch.stack(each_negative_db_map, 0)  # [nneg,3,h,w]
            positive_db_map.append(each_positive_db_map)
            negative_db_map.append(each_negative_db_map)
        positive_db_map = torch.stack(positive_db_map, 0)  # [nmap,3,h,w]
        negative_db_map = torch.stack(negative_db_map, 1)  # [nneg,nmap,3,h,w]


        positive_db_eastnorth = torch.tensor([self.database_infos[best_positive_index]['east'], self.database_infos[best_positive_index]['north']])
        negative_db_eastnorth = [torch.tensor([self.database_infos[e]['east'], self.database_infos[e]['north']]) for e in neg_indexes]
        negative_db_eastnorth = torch.stack(negative_db_eastnorth, 0)  # [nneg,2]


        # negative_db_map = torch.stack(negative_db_map, 0)  # [nneg,3,h,w]
        db_map = torch.cat((positive_db_map.unsqueeze(0), negative_db_map), 0)  # [1+nneg,nmap,3,h,w]
        db_eastnorth = torch.cat((positive_db_eastnorth.unsqueeze(0), negative_db_eastnorth), 0)  # [1+nneg,2]


        triplets_local_indexes = torch.empty((0, 3), dtype=torch.int)
        for neg_num in range(len(neg_indexes)):
            triplets_local_indexes = torch.cat((triplets_local_indexes, torch.tensor([0, 1, 2 + neg_num]).reshape(1, 3)))
        assert query_index < self.queries_num   

        output_dict = {
            'query_image': query_image,
            'query_sph': query_sph,
            'query_bev': query_bev,
            'query_pc': query_pc,
            'query_pc_rgb': query_pc_rgb,
            'query_pc_normal': query_pc_normal,
            'query_eastnorth': query_eastnorth,
            'db_map': db_map,
            'db_eastnorth': db_eastnorth
        }

        return output_dict, triplets_local_indexes, self.triplets_global_indexes[index]
    
    def __len__(self):
        if self.is_inference:
            # At inference time return the number of images. This is used for caching or computing NetVLAD's clusters
            length = super().__len__()
            return length
        else:
            length = len(self.triplets_global_indexes)
            return length
    
    def compute_triplets(self, args, model, modelq=None):
        self.is_inference = True
        if self.mining == "full":
            self.compute_triplets_full(args, model)
        elif self.mining == "partial" or self.mining == "msls_weighted":
            self.compute_triplets_partial(args, model)
        elif self.mining == "random":
            self.compute_triplets_random(args, model)
        elif self.mining == 'partial_sep':
            assert modelq is not None
            self.compute_triplets_partial_sep(args, model, modelq)
        else:
            raise NotImplementedError
    
    @staticmethod
    def compute_cache(args, model, subset_ds, cache_shape):
        """Compute the cache containing features of images, which is used to
        find best positive and hardest negatives."""
        subset_dl = DataLoader(dataset=subset_ds, num_workers=args.num_workers,
                               batch_size=args.infer_batch_size, shuffle=False,
                               pin_memory=(args.device == "cuda"))
        model = model.eval()
        
        # RAMEfficient2DMatrix can be replaced by np.zeros, but using
        # RAMEfficient2DMatrix is RAM efficient for full database mining.
        cache = RAMEfficient2DMatrix(cache_shape, dtype=np.float32) # [db+q, c]
        with torch.no_grad():
            for images, indexes in tqdm(subset_dl):
                images = images.to(args.device)
                features = model(images)
                cache[indexes.numpy()] = features.cpu().numpy()
        return cache
    


    # =================== compute cache separate
    @staticmethod
    def compute_cache_sep(args, model, subset_ds, cache_shape, modelq):
        """Compute the cache containing features of images, which is used to
        find best positive and hardest negatives."""
        # subset_dl = DataLoader(dataset=subset_ds, num_workers=args.num_workers,
        #                        batch_size=args.infer_batch_size, shuffle=False,
        #                        pin_memory=(args.device == "cuda"))
        model = model.eval()
        modelq = modelq.eval()

        subset_ds_db = Subset(subset_ds.dataset, list(range(subset_ds.dataset.database_num)))
        subset_ds_q = Subset(subset_ds.dataset, list(range(subset_ds.dataset.database_num, len(subset_ds.dataset))))
        subset_dl_db = DataLoader(dataset=subset_ds_db, num_workers=args.num_workers,
                                 batch_size=args.infer_batch_size, shuffle=False,
                                 pin_memory=(args.device == "cuda"),
                                 collate_fn=kitti360_collate_fn_cache_db
                                 )
        subset_dl_q = DataLoader(dataset=subset_ds_q, num_workers=args.num_workers,
                                batch_size=args.infer_batch_size, shuffle=False,
                                pin_memory=(args.device == "cuda"),
                                collate_fn=kitti360_collate_fn_cache_q
                                )
        # RAMEfficient2DMatrix can be replaced by np.zeros, but using
        # RAMEfficient2DMatrix is RAM efficient for full database mining.
        cache = RAMEfficient2DMatrix(cache_shape, dtype=np.float32) # [db+q, c]
        import contextlib
        amp_dt = getattr(args, 'amp_dtype', 'none')
        if amp_dt == 'bf16':
            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        elif amp_dt == 'fp16':
            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
        else:
            amp_ctx = contextlib.nullcontext()
        with torch.no_grad(), amp_ctx:
            # db
            for data_dict, indexes in tqdm(subset_dl_db):
                data_dict = {k: v.to(args.device) for k, v in data_dict.items()}
                features = model(data_dict, mode='db')
                cache[indexes.numpy()] = features['embedding'].float().cpu().numpy()
            # q
            for data_dict, indexes in tqdm(subset_dl_q):
                data_dict = {k: v.to(args.device) for k, v in data_dict.items()}
                features = modelq(data_dict, mode='q')
                cache[indexes.numpy()] = features['embedding'].float().cpu().numpy()
        return cache
    



    def get_query_features(self, query_index, cache):
        query_features = cache[query_index + self.database_num]
        if query_features is None:
            raise RuntimeError(f"For query {self.queries_paths[query_index]} " +
                               f"with index {query_index} features have not been computed!\n" +
                               "There might be some bug with caching")
        return query_features
    
    def get_best_positive_index(self, args, query_index, cache, query_features):
        positives_features = cache[self.hard_positives_per_query[query_index]]
        faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(positives_features)
        # Search the best positive (within 10 meters AND nearest in features space)
        _, best_positive_num = faiss_index.search(query_features.reshape(1, -1), 1)
        best_positive_index = self.hard_positives_per_query[query_index][best_positive_num[0]].item()
        return best_positive_index
    
    def get_hardest_negatives_indexes(self, args, cache, query_features, neg_samples):
        neg_features = cache[neg_samples]
        faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(neg_features)
        # Search the 10 nearest negatives (further than 25 meters and nearest in features space)
        _, neg_nums = faiss_index.search(query_features.reshape(1, -1), self.negs_num_per_query)
        neg_nums = neg_nums.reshape(-1)
        neg_indexes = neg_samples[neg_nums].astype(np.int32)
        return neg_indexes
    
    def compute_triplets_random(self, args, model):
        self.triplets_global_indexes = []
        # Take 1000 random queries
        sampled_queries_indexes = np.random.choice(self.queries_num, args.cache_refresh_rate, replace=False)
        # Take all the positives
        positives_indexes = [self.hard_positives_per_query[i] for i in sampled_queries_indexes]
        positives_indexes = [p for pos in positives_indexes for p in pos]  # Flatten list of lists to a list
        positives_indexes = list(np.unique(positives_indexes))
        
        # Compute the cache only for queries and their positives, in order to find the best positive
        subset_ds = Subset(self, positives_indexes + list(sampled_queries_indexes + self.database_num))
        cache = self.compute_cache(args, model, subset_ds, (len(self), args.features_dim))
        
        # This loop's iterations could be done individually in the __getitem__(). This way is slower but clearer (and yields same results)
        for query_index in tqdm(sampled_queries_indexes):
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)
            
            # Choose some random database images, from those remove the soft_positives, and then take the first 10 images as neg_indexes
            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = np.random.choice(self.database_num, size=self.negs_num_per_query+len(soft_positives), replace=False)
            neg_indexes = np.setdiff1d(neg_indexes, soft_positives, assume_unique=True)[:self.negs_num_per_query]
            
            self.triplets_global_indexes.append((query_index, best_positive_index, *neg_indexes))
        # self.triplets_global_indexes is a tensor of shape [1000, 12]
        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes)
    
    def compute_triplets_full(self, args, model):
        self.triplets_global_indexes = []
        # Take 1000 random queries
        sampled_queries_indexes = np.random.choice(self.queries_num, args.cache_refresh_rate, replace=False)
        # Take all database indexes
        database_indexes = list(range(self.database_num))
        #  Compute features for all images and store them in cache
        subset_ds = Subset(self, database_indexes + list(sampled_queries_indexes + self.database_num))
        cache = self.compute_cache(args, model, subset_ds, (len(self), args.features_dim))
        
        # This loop's iterations could be done individually in the __getitem__(). This way is slower but clearer (and yields same results)
        for query_index in tqdm(sampled_queries_indexes):
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)
            # Choose 1000 random database images (neg_indexes)
            neg_indexes = np.random.choice(self.database_num, self.neg_samples_num, replace=False)
            # Remove the eventual soft_positives from neg_indexes
            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = np.setdiff1d(neg_indexes, soft_positives, assume_unique=True)
            # Concatenate neg_indexes with the previous top 10 negatives (neg_cache)
            neg_indexes = np.unique(np.concatenate([self.neg_cache[query_index], neg_indexes]))
            # Search the hardest negatives
            neg_indexes = self.get_hardest_negatives_indexes(args, cache, query_features, neg_indexes)
            # Update nearest negatives in neg_cache
            self.neg_cache[query_index] = neg_indexes
            self.triplets_global_indexes.append((query_index, best_positive_index, *neg_indexes))
        # self.triplets_global_indexes is a tensor of shape [1000, 12]
        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes)
    



    # =================== partial

    def compute_triplets_partial(self, args, model):
        self.triplets_global_indexes = []
        # Take 1000 random queries
        if self.mining == "partial":
            sampled_queries_indexes = np.random.choice(self.queries_num, args.cache_refresh_rate, replace=False) 
        elif self.mining == "msls_weighted":  # Pick night and sideways queries with higher probability
            sampled_queries_indexes = np.random.choice(self.queries_num, args.cache_refresh_rate, replace=False, p=self.weights)
        
        # Sample 1000 random database images for the negatives
        sampled_database_indexes = np.random.choice(self.database_num, self.neg_samples_num, replace=False) # why need this step?
        # Take all the positives
        positives_indexes = [self.hard_positives_per_query[i] for i in sampled_queries_indexes] # [array, array, ...]
        positives_indexes = [p for pos in positives_indexes for p in pos] # [int, int, ...]
        # Merge them into database_indexes and remove duplicates
        database_indexes = list(sampled_database_indexes) + positives_indexes
        database_indexes = list(np.unique(database_indexes))
        
        # self is inference True length = 23949
        subset_ds = Subset(self, database_indexes + list(sampled_queries_indexes + self.database_num))
        # [partial dataset + partial query] cache
        # query_index = query_index + self.database_num
        cache_length = len(self)
        cache = self.compute_cache(args, model, subset_ds, cache_shape=(cache_length, args.features_dim)) 
        
        # This loop's iterations could be done individually in the __getitem__(). This way is slower but clearer (and yields same results)
        for query_index in tqdm(sampled_queries_indexes):
            query_features = self.get_query_features(query_index, cache) # query_features = cache[query_index + self.database_num]
            best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)
            
            # Choose the hardest negatives within sampled_database_indexes, ensuring that there are no positives
            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = np.setdiff1d(sampled_database_indexes, soft_positives, assume_unique=True)
            
            # Take all database images that are negatives and are within the sampled database images (aka database_indexes)
            neg_indexes = self.get_hardest_negatives_indexes(args, cache, query_features, neg_indexes)
            self.triplets_global_indexes.append((query_index, best_positive_index, *neg_indexes))
        # self.triplets_global_indexes is a tensor of shape [1000, 12]
        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes)



    # =================== partial_sep 

    def compute_triplets_partial_sep(self, args, model, modelq):
        self.triplets_global_indexes = []
        # Take 1000 random queries
        if self.mining in ["partial", 'partial_sep']:
            sampled_queries_indexes = np.random.choice(self.queries_num, args.cache_refresh_rate, replace=False) 
        elif self.mining == "msls_weighted":  # Pick night and sideways queries with higher probability
            sampled_queries_indexes = np.random.choice(self.queries_num, args.cache_refresh_rate, replace=False, p=self.weights)
        
        # Sample 1000 random database images for the negatives
        sampled_database_indexes = np.random.choice(self.database_num, self.neg_samples_num, replace=False) # why need this step?
        # Take all the positives
        positives_indexes = [self.hard_positives_per_query[i] for i in sampled_queries_indexes] # [array, array, ...]
        positives_indexes = [p for pos in positives_indexes for p in pos] # [int, int, ...]
        # Merge them into database_indexes and remove duplicates
        database_indexes = list(sampled_database_indexes) + positives_indexes
        database_indexes = list(np.unique(database_indexes))
        
        # self is inference True length = 23949
        subset_ds = Subset(self, database_indexes + list(sampled_queries_indexes + self.database_num))
        # [partial dataset + partial query] cache
        # query_index = query_index + self.database_num
        cache_length = len(self)
        # cache = self.compute_cache(args, model, subset_ds, cache_shape=(cache_length, args.features_dim)) 
        cache = self.compute_cache_sep(args, model, subset_ds, cache_shape=(cache_length, args.features_dim), modelq=modelq) 
        
        # This loop's iterations could be done individually in the __getitem__(). This way is slower but clearer (and yields same results)
        for query_index in tqdm(sampled_queries_indexes): # max-12830  min-16
            query_features = self.get_query_features(query_index, cache) # query_features = cache[query_index + self.database_num]
            best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)
            
            # Choose the hardest negatives within sampled_database_indexes, ensuring that there are no positives
            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = np.setdiff1d(sampled_database_indexes, soft_positives, assume_unique=True)
            
            # Take all database images that are negatives and are within the sampled database images (aka database_indexes)
            neg_indexes = self.get_hardest_negatives_indexes(args, cache, query_features, neg_indexes)
            self.triplets_global_indexes.append((query_index, best_positive_index, *neg_indexes))
        # self.triplets_global_indexes is a tensor of shape [1000, 12]
        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes)









class RAMEfficient2DMatrix:
    """This class behaves similarly to a numpy.ndarray initialized
    with np.zeros(), but is implemented to save RAM when the rows
    within the 2D array are sparse. In this case it's needed because
    we don't always compute features for each image, just for few of
    them"""
    def __init__(self, shape, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        self.matrix = [None] * shape[0]
    
    def __setitem__(self, indexes, vals):
        assert vals.shape[1] == self.shape[1], f"{vals.shape[1]} {self.shape[1]}"
        for i, val in zip(indexes, vals):
            self.matrix[i] = val.astype(self.dtype, copy=False)
    
    def __getitem__(self, index):
        if hasattr(index, "__len__"):
            return np.array([self.matrix[i] for i in index])
        else:
            return self.matrix[index]
