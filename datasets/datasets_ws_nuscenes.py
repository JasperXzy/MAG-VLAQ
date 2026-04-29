
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

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import open3d as o3d


# Utonia official input pipeline constants (transform.default(scale=0.2)
# + GridSample(grid_size=0.01)). Effective voxel = 0.01 / 0.2 = 0.05 m.
UTONIA_GLOBAL_SCALE = 0.2
UTONIA_GRID_SIZE = 0.01

_DATASET_PROGRESS_CALLBACK = None


def set_progress_callback(callback):
    global _DATASET_PROGRESS_CALLBACK
    _DATASET_PROGRESS_CALLBACK = callback


def _progress(iterable, args=None, desc=None, disable=False):
    if _DATASET_PROGRESS_CALLBACK is not None and not disable:
        total = len(iterable) if hasattr(iterable, "__len__") else None
        _DATASET_PROGRESS_CALLBACK("start", desc, total)
        try:
            for item in iterable:
                yield item
                _DATASET_PROGRESS_CALLBACK("advance", desc, 1)
        finally:
            _DATASET_PROGRESS_CALLBACK("close", desc, None)
        return

    disable = disable or bool(getattr(args, "disable_dataset_tqdm", False))
    yield from tqdm(
        iterable,
        desc=desc,
        disable=disable,
        dynamic_ncols=True,
        leave=False,
        mininterval=1.0,
    )


def estimate_normals_o3d(pts, k=30):
    """Per-point normal estimation with normals oriented toward LiDAR origin."""
    n = pts.shape[0]
    if n < 3:
        return np.zeros((n, 3), dtype=np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=min(k, n))
    )
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))
    return np.asarray(pcd.normals, dtype=np.float32)


def _rand_rotation_outdoor_scene():
    theta_z = 2.0 * np.pi * (np.random.rand() - 0.5)
    theta_x = (np.pi / 16.0) * 2.0 * (np.random.rand() - 0.5)
    theta_y = (np.pi / 16.0) * 2.0 * (np.random.rand() - 0.5)
    cz, sz = np.cos(theta_z), np.sin(theta_z)
    cx, sx = np.cos(theta_x), np.sin(theta_x)
    cy, sy = np.cos(theta_y), np.sin(theta_y)
    Rz = np.array([[cz, -sz, 0.0], [sz,  cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    Ry = np.array([[cy,  0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


def _rand_scale_jitter(eta=1.1):
    log_eta = np.log(eta)
    return float(np.exp(np.random.uniform(-log_eta, log_eta)))


def _rand_anisotropic_jitter(eta=1.1):
    log_eta = np.log(eta)
    return np.exp(np.random.uniform(-log_eta, log_eta, size=3)).astype(np.float32)


def _causal_modality_blind(rgb, normal, p_drop_rgb=0.3, p_drop_normal=0.3):
    if np.random.rand() < p_drop_rgb:
        rgb = np.zeros_like(rgb)
    if np.random.rand() < p_drop_normal:
        normal = np.zeros_like(normal)
    return rgb, normal


def _dedup_grid(coords_int_np):
    _, keep = np.unique(coords_int_np, axis=0, return_index=True)
    keep.sort()
    return keep


def _build_utonia_batch(query_pc, query_pc_rgb, query_pc_normal, train):
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

        if train:
            R = _rand_rotation_outdoor_scene()
            s = _rand_scale_jitter(eta=1.1)
            aniso = _rand_anisotropic_jitter(eta=1.1)
            p_aug = (p_np @ R) * (s * aniso)
            n_aug = n_np @ R
            r_np, n_aug = _causal_modality_blind(r_np, n_aug, p_drop_rgb=0.3, p_drop_normal=0.3)
        else:
            p_aug = p_np
            n_aug = n_np

        p_scaled = p_aug * UTONIA_GLOBAL_SCALE
        p_int = np.floor(p_scaled / UTONIA_GRID_SIZE).astype(np.int32)
        keep = _dedup_grid(p_int)
        p_scaled_k = p_scaled[keep]
        p_int_k = p_int[keep]
        r_k = r_np[keep] if r_np.shape[0] == p_np.shape[0] else np.zeros((len(keep), 3), dtype=np.float32)
        n_k = n_aug[keep]

        n_pts = len(keep)
        utonia_coord_list.append(torch.from_numpy(p_scaled_k))
        utonia_grid_list.append(torch.from_numpy(p_int_k))
        utonia_rgb_list.append(torch.from_numpy(r_k))
        utonia_normal_list.append(torch.from_numpy(n_k))
        utonia_point_counts.append(n_pts)

        me_int = np.floor(p_aug).astype(np.int32)
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
    return coords, feats, utonia_coord, utonia_grid_coord, utonia_feat, utonia_rgb, utonia_normal, utonia_offset








# trainselectlocationlist = [  # define which location to use
#     "2013_05_28_drive_0000_sync",
#     "2013_05_28_drive_0002_sync",
#     "2013_05_28_drive_0003_sync",
#     "2013_05_28_drive_0004_sync",
#     "2013_05_28_drive_0005_sync",
#     "2013_05_28_drive_0006_sync",
#     "2013_05_28_drive_0007_sync",
#     "2013_05_28_drive_0009_sync",
#     "2013_05_28_drive_0010_sync",
# ]

# testselectlocationlist = [
#     "2013_05_28_drive_0000_sync",
#     "2013_05_28_drive_0002_sync",
#     "2013_05_28_drive_0003_sync",
#     "2013_05_28_drive_0004_sync",
#     "2013_05_28_drive_0005_sync",
#     "2013_05_28_drive_0006_sync",
#     "2013_05_28_drive_0007_sync",
#     "2013_05_28_drive_0009_sync",
#     "2013_05_28_drive_0010_sync",
# ]


trainselectlocationlist = [
    'singapore-onenorth',
    'singapore-hollandvillage',
    'singapore-queenstown',
    'boston-seaport'
]

testselectlocationlist = [
    'singapore-onenorth',
    'singapore-hollandvillage',
    'singapore-queenstown',
    'boston-seaport'
]




base_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def path_to_pil_img(path):
    image = Image.open(path)
    image = image.convert("RGB")
    return image




def nuscenes_collate_fn(batch):
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
    # pcs = [e[0]['pc'] for e in batch]
    query_pc = [e[0]['query_pc'] for e in batch]
    query_image = torch.stack([e[0]['query_image'] for e in batch]) 
    query_bev = torch.stack([e[0]['query_bev'] for e in batch])
    query_sph = torch.stack([e[0]['query_sph'] for e in batch])
    query_eastnorth = torch.stack([e[0]['query_eastnorth'] for e in batch])
    db_map = torch.stack([e[0]['db_map'] for e in batch])
    db_eastnorth = torch.stack([e[0]['db_eastnorth'] for e in batch])
    # positive_db_map = torch.stack([e[0]['positive_db_map'] for e in batch]) 
    # negative_db_maps = torch.stack([e[0]['negative_db_maps'] for e in batch]) 
    query_pc_rgb = [e[0]['query_pc_rgb'] for e in batch]
    query_pc_normal = [e[0]['query_pc_normal'] for e in batch]
    (
        coords,
        feats,
        utonia_coord,
        utonia_grid_coord,
        utonia_feat,
        utonia_rgb,
        utonia_normal,
        utonia_offset,
    ) = _build_utonia_batch(query_pc, query_pc_rgb, query_pc_normal, train=True)

    # feats = torch.ones([coords.shape[0], 1]).float()
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
        'utonia_coord': utonia_coord,
        'utonia_grid_coord': utonia_grid_coord,
        'utonia_feat': utonia_feat,
        'utonia_rgb': utonia_rgb,
        'utonia_normal': utonia_normal,
        'utonia_offset': utonia_offset,
    }
    # return images, torch.cat(tuple(triplets_local_indexes)), triplets_global_indexes
    return output_dict, torch.cat(tuple(triplets_local_indexes)), triplets_global_indexes





def nuscenes_collate_fn_cache_db(batch):
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
    # query_sph = torch.stack([e[0]['query_sph'] for e in batch])
    db_map = torch.stack([e[0]['db_map'] for e in batch])
    indices = torch.tensor([e[1] for e in batch])
    db_location = [e[0]['db_location'] for e in batch]
    db_eastnorth = torch.stack([e[0]['db_eastnorth'] for e in batch])

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
        # 'query_sph': query_sph,
        'db_map': db_map,
        'db_location': db_location,
        'db_eastnorth': db_eastnorth,
    }
    return output_dict, indices




def nuscenes_collate_fn_cache_q(batch):
    '''
    output of collate_fn should be applicable with .to(device)
    '''
    # images = torch.stack([e[0]['image'] for e in batch]) # [12,c,h,w] -> [b*12,c,h,w]
    # bevs = torch.stack([e[0]['bev'] for e in batch]) # [3,h,w] -> [b,3,h,w]
    # # sphs = torch.stack([e[0]['sph'] for e in batch]) # [3,h,w] -> [b,3,h,w]
    # pcs = [e[0]['pc'] for e in batch]
    query_image = torch.stack([e[0]['query_image'] for e in batch])
    # query_image = query_image * 0
    query_bev = torch.stack([e[0]['query_bev'] for e in batch])
    query_sph = torch.stack([e[0]['query_sph'] for e in batch])
    query_pc = [e[0]['query_pc'] for e in batch]
    query_pc_rgb = [e[0]['query_pc_rgb'] for e in batch]
    query_pc_normal = [e[0]['query_pc_normal'] for e in batch]
    (
        coords,
        feats,
        utonia_coord,
        utonia_grid_coord,
        utonia_feat,
        utonia_rgb,
        utonia_normal,
        utonia_offset,
    ) = _build_utonia_batch(query_pc, query_pc_rgb, query_pc_normal, train=False)
    query_location = [e[0]['query_location'] for e in batch]
    query_eastnorth = torch.stack([e[0]['query_eastnorth'] for e in batch])

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
        'query_location': query_location,
        'query_eastnorth': query_eastnorth,
        'utonia_coord': utonia_coord,
        'utonia_grid_coord': utonia_grid_coord,
        'utonia_feat': utonia_feat,
        'utonia_rgb': utonia_rgb,
        'utonia_normal': utonia_normal,
        'utonia_offset': utonia_offset,
    }
    return output_dict, indices







# def load_qimage(datapath, split):
#     image = Image.open(datapath)
#     image = image.convert('RGB')
#     if split == 'train':
#         tf = TVT.Compose([TVT.Resize(args.q_resize), 
#                         # TVT.ColorJitter(brightness=args.q_jitter, contrast=args.q_jitter, saturation=args.q_jitter, hue=min(0.5, args.q_jitter)),
#                         TVT.ToTensor(),
#                           TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#                         # TVT.Normalize(mean=0.5, std=0.22)
#                         ])
#     elif split == 'test':
#         tf = TVT.Compose([TVT.Resize(args.q_resize), 
#                         TVT.ToTensor(),
#                           TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#                         # TVT.Normalize(mean=0.5, std=0.22)
#                         ])
#     image = tf(image)
#     return image



def load_dbimage(datapath, split, args):
    image = Image.open(datapath)
    image = image.convert('RGB')
    if split == 'train':
        tf = TVT.Compose([
            # TVT.CenterCrop(args.db_cropsize),
            TVT.Resize((args.db_resize, args.db_resize), antialias=True),
            # TVT.ColorJitter(brightness=args.db_jitter, contrast=args.db_jitter, saturation=args.db_jitter, hue=min(0.5, args.db_jitter)),
            TVT.ToTensor(),
            TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # TVT.Normalize(mean=0.5, std=0.22)
                        ])
    elif split == 'test':
        tf = TVT.Compose([
            # TVT.CenterCrop(args.db_cropsize),
            TVT.Resize((args.db_resize, args.db_resize), antialias=True),
            TVT.ToTensor(),
            TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # TVT.Normalize(mean=0.5, std=0.22)
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




def generate_sph_from_pc(pc, w=361, h=61, args=None):
    # kitti   w=361  h=61
    # ithaca  w=361  h=101
    # kitti360 w=361 h=61
    # w = 361
    # h = 61
    # if 'ithaca365' in args.dataset_name:
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
    if args is not None and 'ithaca365' in args.dataset_name:
        ids_h = (uv[:,0] < h) &  (uv[:,0] >= 0)
        uv = uv[ids_h]
        r = r[ids_h]
    sph = np.zeros([h, w])
    sph[uv[:,0], uv[:,1]] = r
    # if 'ithaca365' in args.dataset_name:
    #     sph = sph[25:80]
    # plt.imshow(sph)
    # plt.imsave('sph.png', sph)
    # plt.close()
    # plt.show()
    return sph
    



def load_pc_bev(file_path, split, args): # filename is the same as load_pc
    pc = np.fromfile(file_path, dtype=np.float32).reshape(-1,3) # for kitti360_voxel
    bev = generate_bev_from_pc(pc, w=200, max_thd=100)
    # ==== bev
    bev = Image.fromarray(bev).convert('RGB')
    # bev.show()
    (_w,_h) = bev.size
    assert args.bev_resize <= 1
    resize_ratio = random.uniform(args.bev_resize, 2-args.bev_resize)
    resize_size = int(resize_ratio*min(_w,_h))
    assert resize_size >= args.bev_cropsize

    if args.bev_resize_mode == 'nearest':
        resize_mode = InterpolationMode.NEAREST
    elif args.bev_resize_mode == 'bilinear':
        resize_mode = InterpolationMode.BILINEAR
    else:
        raise NotImplementedError
    
    if args.bev_rotate_mode == 'nearest':
        rotate_mode = InterpolationMode.NEAREST
    elif args.bev_rotate_mode == 'bilinear':
        rotate_mode = InterpolationMode.BILINEAR
    else:
        raise NotImplementedError
    
    if split == 'train':
        tf = TVT.Compose([
            TVT.Resize(resize_size, interpolation=resize_mode, antialias=True),
            TVT.RandomRotation(args.bev_rotate, interpolation=rotate_mode),
            TVT.CenterCrop(args.bev_cropsize),
            TVT.ColorJitter(brightness=args.bev_jitter, contrast=args.bev_jitter, saturation=args.bev_jitter, hue=min(0.5, args.bev_jitter)),
            TVT.ToTensor(),
            TVT.Normalize(mean=args.bev_mean, std=args.bev_std)
        ])
    elif split == 'test':
        tf = TVT.Compose([
            TVT.Resize(resize_size, interpolation=resize_mode, antialias=True),
            TVT.CenterCrop(args.bev_cropsize),
            TVT.ToTensor(),
            TVT.Normalize(mean=args.bev_mean, std=args.bev_std)
        ])
    else:
        raise NotImplementedError
    bev = tf(bev)
    # # ---- DEBUG:viz
    # bev = bev*args.bev_std + args.bev_mean
    # bev = TVT.ToPILImage()(bev)
    # bev.show()
    return pc, bev



def load_pc_sph(file_path, split, args): # filename is the same as load_pc
    pc = np.fromfile(file_path, dtype=np.float32).reshape(-1,3) # for kitti360_voxel
    sph = generate_sph_from_pc(pc, w=361, h=61, args=args)
    # ==== sph
    sph = Image.fromarray(sph).convert('RGB')
    (_w,_h) = sph.size
    assert args.sph_resize <= 1
    resize_ratio = random.uniform(args.sph_resize, 2-args.sph_resize)
    resize_size = int(resize_ratio*min(_w,_h))
    if split == 'train':
        tf = TVT.Compose([
            TVT.Resize(resize_size, interpolation=InterpolationMode.NEAREST, antialias=True),
            TVT.ColorJitter(brightness=args.sph_jitter, contrast=args.sph_jitter, saturation=args.sph_jitter, hue=min(0.5, args.sph_jitter)),
            TVT.ToTensor(),
            TVT.Normalize(mean=args.sph_mean, std=args.sph_std)
        ])
    elif split == 'test':
        tf = TVT.Compose([
            TVT.Resize(resize_size, interpolation=InterpolationMode.NEAREST, antialias=True),
            TVT.ToTensor(),
            TVT.Normalize(mean=args.sph_mean, std=args.sph_std)
        ])
    else:
        raise NotImplementedError
    sph = tf(sph)
    return pc, sph














def get_location_from_sample_token(nusc, sample_token):
    sample = nusc.get('sample', sample_token)
    scene_token = sample['scene_token']
    location = nusc.get('log', nusc.get('scene', scene_token)['log_token'])['location']
    return location






def get_latloneastnorth_from_sample_token(nusc, sample_token, location):
    sample = nusc.get('sample', sample_token)
    ego_pose = nusc.get('ego_pose', sample['data']['LIDAR_TOP'])
    ego_pose = ego_pose['translation']
    ego_pose = np.array(ego_pose)
    if location == 'boston-seaport':
        eastingorg, northingorg, zone_number, zone_letter = utm.from_latlon(42.336849169438615, -71.05785369873047)
        # clock-wise rotate
        degree = 1.5
        R = np.array([[np.cos(np.pi/180*degree), -np.sin(np.pi/180*degree)], 
                      [np.sin(np.pi/180*degree), np.cos(np.pi/180*degree)]])
        ego_pose[0:2] = ego_pose[0:2] @ R 
    elif location == 'singapore-onenorth':
        eastingorg, northingorg, zone_number, zone_letter = utm.from_latlon(1.2882100868743724, 103.78475189208984)
    elif location == 'singapore-hollandvillage':
        eastingorg, northingorg, zone_number, zone_letter = utm.from_latlon(1.2993652317780957, 103.78217697143555)
    elif location == 'singapore-queenstown':
        eastingorg, northingorg, zone_number, zone_letter = utm.from_latlon(1.2782562240223188, 103.76741409301758)
    else:
        raise NotImplementedError
    ego_pose[0] = ego_pose[0] + eastingorg
    ego_pose[1] = ego_pose[1] + northingorg
    easting = ego_pose[0]
    northing = ego_pose[1]
    latlon = utm.to_latlon(easting, northing, zone_number, zone_letter)
    outputdict = {
        'lat': latlon[0],
        'lon': latlon[1],
        'east': easting,
        'north': northing,
        'zone_number': zone_number,
        'zone_letter': zone_letter
    }
    return outputdict






def get_datapaths_from_sample_token(nusc, sample_token):
    sample = nusc.get('sample', sample_token)
    sensornames = ['LIDAR_TOP', 
                   'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    datapaths = {}
    for sensorname in sensornames:
        datatoken = sample['data'][sensorname]
        data = nusc.get('sample_data', datatoken)
        dataname = data['filename']
        datapath = os.path.join(nusc.dataroot, dataname)
        # assert os.path.exists(datapath), f'{datapath} does not exist'
        datapaths[sensorname] = datapath

    return datapaths


def _camnames_from_args(args):
    name_map = {
        'f': 'CAM_FRONT',
        'fl': 'CAM_FRONT_LEFT',
        'fr': 'CAM_FRONT_RIGHT',
        'b': 'CAM_BACK',
        'bl': 'CAM_BACK_LEFT',
        'br': 'CAM_BACK_RIGHT',
    }
    return [name_map[name] for name in args.camnames.split('_')]


def _quat_to_rotmat(rotation):
    q = np.asarray(rotation, dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1e-8)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _transform_points(points, rotation, translation):
    rot = _quat_to_rotmat(rotation)
    trans = np.asarray(translation, dtype=np.float32)
    return points @ rot.T + trans


def _inverse_transform_points(points, rotation, translation):
    rot = _quat_to_rotmat(rotation)
    trans = np.asarray(translation, dtype=np.float32)
    return (points - trans) @ rot


def _lidar_to_global(points, lidar_cs, lidar_pose):
    points = _transform_points(points, lidar_cs['rotation'], lidar_cs['translation'])
    points = _transform_points(points, lidar_pose['rotation'], lidar_pose['translation'])
    return points


def _global_to_camera(points_global, cam_cs, cam_pose):
    points = _inverse_transform_points(points_global, cam_pose['rotation'], cam_pose['translation'])
    points = _inverse_transform_points(points, cam_cs['rotation'], cam_cs['translation'])
    return points


def _load_lidar_points(file_path):
    try:
        points = LidarPointCloud.from_file(file_path).points.T[:, :3]
        return points.astype(np.float32, copy=False)
    except Exception:
        arr = np.fromfile(file_path, dtype=np.float32)
        if arr.size % 5 == 0:
            return arr.reshape(-1, 5)[:, :3].astype(np.float32, copy=False)
        if arr.size % 4 == 0:
            return arr.reshape(-1, 4)[:, :3].astype(np.float32, copy=False)
        if arr.size % 3 == 0:
            return arr.reshape(-1, 3).astype(np.float32, copy=False)
        raise ValueError(f"Cannot parse LiDAR point cloud: {file_path}, float_count={arr.size}")


def _sensor_file_path(nusc, sample, sensorname):
    datatoken = sample['data'][sensorname]
    data = nusc.get('sample_data', datatoken)
    return data, os.path.join(nusc.dataroot, data['filename'])


def _resized_camera_path(raw_path, size=256):
    path_parts = raw_path.split('/')
    path_parts[-2] += f'_size{size}'
    return '/'.join(path_parts)


def colorize_lidar_points_nuscenes(nusc, sample, points, camera_names):
    """Project LiDAR points into selected NuScenes cameras and average RGB."""
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    lidar_data, _ = _sensor_file_path(nusc, sample, 'LIDAR_TOP')
    lidar_cs = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    lidar_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    points_global = _lidar_to_global(points, lidar_cs, lidar_pose)

    rgb_sum = np.zeros((points.shape[0], 3), dtype=np.float32)
    count = np.zeros((points.shape[0],), dtype=np.float32)
    for camera_name in camera_names:
        cam_data, raw_path = _sensor_file_path(nusc, sample, camera_name)
        img_path = raw_path if os.path.exists(raw_path) else _resized_camera_path(raw_path)
        if not os.path.exists(img_path):
            continue

        cam_cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        cam_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
        pts_cam = _global_to_camera(points_global, cam_cs, cam_pose)
        depth = pts_cam[:, 2]
        valid = depth > 0.1
        if not valid.any():
            continue

        intrinsic = np.asarray(cam_cs['camera_intrinsic'], dtype=np.float32)
        pixels = pts_cam @ intrinsic.T
        u = pixels[:, 0] / np.clip(pixels[:, 2], 1e-8, None)
        v = pixels[:, 1] / np.clip(pixels[:, 2], 1e-8, None)

        image = Image.open(img_path).convert('RGB')
        img_w, img_h = image.size
        orig_w = float(cam_data.get('width') or img_w)
        orig_h = float(cam_data.get('height') or img_h)
        if img_w != orig_w or img_h != orig_h:
            u = u * (img_w / orig_w)
            v = v * (img_h / orig_h)

        valid = valid & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
        if not valid.any():
            continue
        image_np = np.asarray(image)
        u_int = np.clip(u[valid].astype(np.int32), 0, img_w - 1)
        v_int = np.clip(v[valid].astype(np.int32), 0, img_h - 1)
        rgb_sum[valid] += image_np[v_int, u_int, :3].astype(np.float32) / 255.0
        count[valid] += 1.0

    rgb = np.zeros((points.shape[0], 3), dtype=np.float32)
    seen = count > 0
    rgb[seen] = rgb_sum[seen] / count[seen, None]
    return rgb







def load_sensordata_from_sampletoken(nusc, sample_token, args):
    sample = nusc.get('sample', sample_token)
    sensornames = ['LIDAR_TOP', 
                   'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    sensordatas = {}
    camera_names = _camnames_from_args(args)
    for sensorname in sensornames:
        datatoken = sample['data'][sensorname]
        data = nusc.get('sample_data', datatoken)
        dataname = data['filename']
        datapath = os.path.join(nusc.dataroot, dataname)
        if sensorname == 'LIDAR_TOP':
            pc = _load_lidar_points(datapath)
            sensordatas['LIDAR_TOP'] = pc
            sensordatas['LIDAR_TOP_RGB'] = colorize_lidar_points_nuscenes(
                nusc, sample, pc, camera_names
            )
            sensordatas['RANGE_DATA'] = torch.empty(0)

            # load bev
            # w = 200
            # max_thd = 100
            # bevdatapath = datapath.replace('.pcd.bin', '.png')
            # bevdatapath = bevdatapath.replace('samples/LIDAR_TOP', f'samples/BEV_DATA_w{w}_thd{max_thd}')
            # bev = Image.open(bevdatapath).convert('RGB')
            # tf = TVT.Compose([TVT.ToTensor(),
            #                 TVT.Resize(256, antialias=True), # [3,32,32]
            #                 ])
            # bev = tf(bev)
            # sensordatas['BEV_DATA'] = bev
            sensordatas['BEV_DATA'] = torch.empty(0)
        else: # for cameras
            datapath = _resized_camera_path(datapath)
            if not os.path.exists(datapath):
                datapath = os.path.join(nusc.dataroot, dataname)
            image = Image.open(datapath)
            tf = TVT.Compose([TVT.Resize((args.q_resize, args.q_resize), antialias=True),
                              TVT.ToTensor(),
                              TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                              ])
            image = tf(image)
            sensordatas[sensorname] = image

    # Stack
    qcam = [] 
    for camera_name in camera_names:
        qcam.append(sensordatas[camera_name])
    # qcam = torch.stack(qcam, dim=0) # [ncam,3,h,w]
    qcam = torch.cat(qcam, dim=-1) # [3,h,w*ncam] panorama cat
    # TODO: support multi-cam setting
    # assert qcam.shape[0] == 1
    # qcam = qcam[0]

    # qcam = torch.stack([sensordatas['CAM_FRONT'], 
    #                     sensordatas['CAM_FRONT_LEFT'], 
    #                     sensordatas['CAM_FRONT_RIGHT'], 
    #                     sensordatas['CAM_BACK'], 
    #                     sensordatas['CAM_BACK_LEFT'], 
    #                     sensordatas['CAM_BACK_RIGHT']], dim=0) # [ncam,3,h,w]

    return (
        qcam,
        sensordatas['RANGE_DATA'],
        sensordatas['BEV_DATA'],
        sensordatas['LIDAR_TOP'],
        sensordatas['LIDAR_TOP_RGB'],
    )



def get_seq_sample_tokens(nusc, sampletoken, seq_len, current_frame_type):
    # current_frame_type: ['new', 'mid', 'old']
    sample = nusc.get('sample', sampletoken)
    seq_sample_tokens = []
    if current_frame_type == 'new':
        for i in range(seq_len):
            seq_sample_tokens = [sample['token']] + seq_sample_tokens
            sampletoken = sample['prev']
            if len(sampletoken) == 0:
                sampletoken = sample['token']
            sample = nusc.get('sample', sampletoken)

    elif current_frame_type == 'old':
        for i in range(seq_len):
            seq_sample_tokens.append(sample['token'])
            sampletoken = sample['next']
            if len(sampletoken) == 0:
                sampletoken = sample['token']
            sample = nusc.get('sample', sampletoken)

    elif current_frame_type == 'mid':
        midsample = sample
        seq_sample_tokens.append(sample['token']) # middle one
        for i in range(seq_len//2):
            sampletoken = sample['prev']
            if len(sampletoken) == 0:
                sampletoken = sample['token']
            sample = nusc.get('sample', sampletoken)
            seq_sample_tokens = [sample['token']] + seq_sample_tokens
        sample = midsample
        for i in range(seq_len//2):
            sampletoken = sample['next']
            if len(sampletoken) == 0:
                sampletoken = sample['token']
            sample = nusc.get('sample', sampletoken)
            seq_sample_tokens.append(sample['token'])

    assert len(seq_sample_tokens) == seq_len
    return seq_sample_tokens










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

class NuScenesBaseDataset(data.Dataset):
    """Dataset with images from database and queries, used for inference (testing and building cache).
    """
    def __init__(self, args, datasets_folder="datasets", dataset_name="pitts30k", split="train"):
        super().__init__()
        self.args = args
        self.dataset_name = dataset_name
        
        self.resize = self.args.resize
        self.test_method = self.args.test_method
        self.split = split
        

        # ==========
        # Use  v1.0-trainval_singapore-onenorth  training
        dataroot = self.args.dataroot
        if split == 'train':
            selectlocationlist = trainselectlocationlist
            selectversion = 'v1.0-trainval'
        elif split == 'test':
            selectlocationlist = testselectlocationlist
            selectversion = 'v1.0-test'
        else:
            raise NotImplementedError
        
        self.nusc = NuScenes(version=selectversion, 
                             dataroot=dataroot, 
                             verbose=False)

        self.queries_infos = []
        self.queries_utms = [] 
        self.location_count = {
            'singapore-onenorth': 0,
            'singapore-hollandvillage': 0,
            'singapore-queenstown': 0,
            'boston-seaport': 0,
        }
        for isample, sample in enumerate(self.nusc.sample):
            if split == 'train':
                if isample % self.args.traindownsample != 0: # using 1/2 samples for training
                    continue
            elif split == 'test': # using all samples for testing
                None
            else:
                raise NotImplementedError
            sampletoken = sample['token']
            location = get_location_from_sample_token(self.nusc, sampletoken)
            if location in selectlocationlist:
                latloneastnorth = get_latloneastnorth_from_sample_token(self.nusc, sampletoken, location)
                datapaths = get_datapaths_from_sample_token(self.nusc, sampletoken)
                qsampleinfo = {
                    'sampletoken': sampletoken,
                    'nextsampletoken': sample['next'],
                    'location': location,
                    'lat': latloneastnorth['lat'],
                    'lon': latloneastnorth['lon'],
                    'east': latloneastnorth['east'],
                    'north': latloneastnorth['north'],
                }
                qsampleinfo.update(datapaths) 
                self.queries_infos.append(qsampleinfo)
                self.queries_utms.append([latloneastnorth['east'], latloneastnorth['north']])
                self.location_count[location] += 1
        self.queries_utms = np.array(self.queries_utms, dtype=np.float32)
        for location in self.location_count.keys():
            print(f"Number of q samples in {selectversion}_{split}_{location}: {self.location_count[location]}")
        print(f"Number of q samples in {selectversion}_{split}_{selectlocationlist}: {len(self.queries_infos)}")
        
        

        # for selectlocation in selectlocationlist:
        #     qpcdir = os.path.join(dataroot, 'data_3d_voxel0.5', selectlocation, 'velodyne_points/data')
        #     qposedir = os.path.join(dataroot, 'data_poses', selectlocation, 'oxts/data')
        #     qimage02dir = os.path.join(dataroot, 'data_2d_raw', selectlocation, 'image_02/data_rgb')
        #     qimage03dir = os.path.join(dataroot, 'data_2d_raw', selectlocation, 'image_03/data_rgb')
        #     qimage0203dir = os.path.join(dataroot, 'data_2d_cat0203', selectlocation, 'image_0203/data_rgb')
        #     qimage0203names = sorted(os.listdir(qimage0203dir))
        #     if split == 'train':
        #         qimage0203names = qimage0203names[:int(len(qimage0203names)*train_ratio)]
        #     elif split == 'test':
        #         qimage0203names = qimage0203names[int(len(qimage0203names)*train_ratio):]
        #     print(f"Number of q samples in {selectlocation}: {len(qimage0203names)}")
        #     for i_sample, qimage0203name in enumerate(qimage0203names):
        #         if i_sample % self.args.traindownsample != 0: # downsample to accelerate training
        #             continue
        #         qpcpath = os.path.join(qpcdir, qimage0203name.replace('.png','.bin'))
        #         qimage02path = os.path.join(qimage02dir, qimage0203name.replace('.png','.png'))
        #         qimage03path = os.path.join(qimage03dir, qimage0203name.replace('.png','.png'))
        #         qimage0203path = os.path.join(qimage0203dir, qimage0203name.replace('.png','.png'))
        #         qposepath = os.path.join(qposedir, qimage0203name.replace('.png','.txt'))
        #         if not os.path.exists(qpcpath): continue
        #         if not os.path.exists(qposepath): continue
        #         qpose = open(qposepath).readline().split(' ')
        #         lat, lon = float(qpose[0]), float(qpose[1])
        #         east, north, _, _ = utm.from_latlon(lat, lon)
        #         qsampleinfo = {
        #             'lat': lat,
        #             'lon': lon,
        #             'east': east,
        #             'north': north,
        #             'qposepath': qposepath,
        #             'qimage02path': qimage02path,
        #             'qimage03path': qimage03path,
        #             'qimage0203path': qimage0203path,
        #             'qpcpath': qpcpath,
        #             'location': selectlocation,
        #         }
        #         self.queries_infos.append(qsampleinfo)
        #         self.queries_utms.append([east, north])
        # self.queries_utms = np.array(self.queries_utms, dtype=np.float32)
        # print(f"Number of q samples in {split}: {len(self.queries_infos)}")



        # Merge all db
        self.database_utms = []
        self.database_infos = []
        scale = 1
        zoom = 20   # higher is closer
        size = 320
        # maptype = self.args.maptype
        # maptype = 'satellite'
        # maptype = 'osm'

        # for selectlocation in selectlocationlist:
        #     dbdir = os.path.join(dataroot, f'aerial_{selectversion}_{selectlocation}_{scale}_{zoom}_{size}_{maptype}')
        #     dbnames = os.listdir(dbdir)
        #     dbnames = sorted(dbnames)
        #     dbeastnorths = [e.split('@')[1:3] for e in dbnames]
        #     self.dbeastnorths += dbeastnorths
        #     dbpaths = [os.path.join(dbdir, dbname) for dbname in dbnames]
        #     self.dbpaths += dbpaths
        #     print(f"Number of aerial db in {selectversion}_{selectlocation}: {len(dbpaths)}")
        # self.dbeastnorths = np.array(self.dbeastnorths, dtype=np.float32)


        for selectlocation in selectlocationlist:
            # db_satellite_dir = os.path.join(dataroot, f'data_aerial_{scale}_{zoom}_{size}_satellite', selectlocation)
            # db_roadmap_dir = os.path.join(dataroot, f'data_aerial_{scale}_{zoom}_{size}_roadmap', selectlocation)
            db_satellite_dir = os.path.join(dataroot, f'aerial_{selectversion}_{selectlocation}_{scale}_{zoom}_{size}_satellite')
            db_roadmap_dir = os.path.join(dataroot, f'aerial_{selectversion}_{selectlocation}_{scale}_{zoom}_{size}_roadmap')
            dbnames = os.listdir(db_satellite_dir)
            dbnames = sorted(dbnames)
            if self.args.share_db == True:
                dbnames = dbnames
            elif self.args.share_db == False:
                dbnames = dbnames
                # if split == 'train':
                #     dbnames = dbnames[:int(len(dbnames)*train_ratio)]
                # elif split == 'test':
                #     dbnames = dbnames[int(len(dbnames)*train_ratio):]
            for i_dbname, dbname in enumerate(dbnames):
                if split == 'train':
                    if i_dbname % self.args.traindownsample != 0: 
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
        knn = NearestNeighbors(n_jobs=self.args.num_workers + 1)
        knn.fit(self.database_utms)
        softposthd = self.args.val_positive_dist_threshold  # 25
        self.soft_positives_per_query = knn.radius_neighbors(self.queries_utms,
                                                             radius=softposthd,
                                                             return_distance=False)
        
        self.database_queries_infos = self.database_infos + self.queries_infos  # db + q
        self.database_num = len(self.database_infos)
        self.queries_num = len(self.queries_infos)
        print(f"Number of db+q: {len(self.database_queries_infos)}")
        a=1

    
    def __getitem__(self, index):
        query_pc_rgb = np.zeros((1, 3), dtype=np.float32)
        query_pc_normal = np.zeros((1, 3), dtype=np.float32)
        if index >= self.database_num: # query
            # query_image = load_qimage(datapath=self.database_queries_infos[index]['qimage0203path'],split=self.split) 
            # query_pc, query_bev = load_pc_bev(file_path=self.database_queries_infos[index]['qpcpath'],split=self.split)
            sampletoken = self.database_queries_infos[index]['sampletoken']
            query_image, query_sph, query_bev, query_pc, query_pc_rgb = load_sensordata_from_sampletoken(self.nusc, sampletoken, self.args)
            query_pc_normal = estimate_normals_o3d(query_pc, k=30)
            query_location = self.database_queries_infos[index]['location']
            query_eastnorth = torch.tensor([self.database_queries_infos[index]['east'], self.database_queries_infos[index]['north']]).float()
            db_location = None
            db_eastnorth = torch.empty(0)
            # query_pc = torch.empty(0)
            db_map = torch.empty(0)
            
        else: # database
            query_image = torch.empty(0)
            query_pc, query_bev, query_sph = torch.empty(0), torch.empty(0), torch.empty(0)
            query_location = None
            query_eastnorth = torch.empty(0)
            maptype = self.args.maptype.split('_')
            db_map = []
            for each_maptype in maptype:
                if each_maptype == 'satellite':
                    each_db_map = load_dbimage(datapath=self.database_queries_infos[index]['db_satellite_path'], split=self.split, args=self.args)
                elif each_maptype == 'roadmap':
                    each_db_map = load_dbimage(datapath=self.database_queries_infos[index]['db_roadmap_path'], split=self.split, args=self.args)
                db_map.append(each_db_map) 
            db_map = torch.stack(db_map, 0) # [nmap,3,h,w]
            db_location = self.database_queries_infos[index]['location']
            db_eastnorth = torch.tensor([self.database_queries_infos[index]['east'], self.database_queries_infos[index]['north']]).float()


        output_dict = {
            'query_image': query_image, 
            'query_bev': query_bev,
            'query_sph': query_sph,
            'query_pc': query_pc,
            'query_pc_rgb': query_pc_rgb,
            'query_pc_normal': query_pc_normal,
            'query_location': query_location,
            'query_eastnorth': query_eastnorth,
            'db_map': db_map,
            'db_location': db_location,
            'db_eastnorth': db_eastnorth,
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

class NuScenesTripletsDataset(NuScenesBaseDataset):
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
        hardposthd = self.args.train_positives_dist_threshold
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
        
        # query_image = load_qimage(datapath=self.queries_infos[query_index]['qimage0203path'],split=self.split) # [3,h,w]
        # query_pc, query_bev = load_pc_bev(file_path=self.queries_infos[query_index]['qpcpath'],split=self.split)
        
        # sampletoken = self.database_queries_infos[query_index]['sampletoken']
        sampletoken = self.queries_infos[query_index]['sampletoken']
        query_image, query_sph, query_bev, query_pc, query_pc_rgb = load_sensordata_from_sampletoken(self.nusc, sampletoken, self.args)
        query_pc_normal = estimate_normals_o3d(query_pc, k=30)
        # query_location = self.queries_infos[query_index]['location']
        query_eastnorth = torch.tensor([self.queries_infos[query_index]['east'], self.queries_infos[query_index]['north']]).float()
        # query_pc = torch.empty(0)


        # positive_db_satellite_map = load_dbimage(datapath=self.database_infos[best_positive_index]['db_satellite_path'],split=self.split)  # [3,h,w]
        # positive_db_roadmap_map = load_dbimage(datapath=self.database_infos[best_positive_index]['db_roadmap_path'],split=self.split)  # [3,h,w]
        # positive_db_map = torch.stack([positive_db_satellite_map, positive_db_roadmap_map], 0)  # [nmap,3,h,w]

        # negative_db_satellite_map = [load_dbimage(datapath=self.database_infos[e]['db_satellite_path'],split=self.split) for e in neg_indexes] # [nneg,3,h,w]
        # negative_db_satellite_map = torch.stack(negative_db_satellite_map, 0)  # [nneg,3,h,w]
        # negative_db_roadmap_map = [load_dbimage(datapath=self.database_infos[e]['db_roadmap_path'],split=self.split) for e in neg_indexes] # [nneg,3,h,w]
        # negative_db_roadmap_map = torch.stack(negative_db_roadmap_map, 0)  # [nneg,3,h,w]
        # negative_db_map = torch.stack([negative_db_satellite_map, negative_db_roadmap_map], 1)  # [nneg,nmap,3,h,w]

        maptype = self.args.maptype.split('_')
        positive_db_map = []
        negative_db_map = []
        for each_maptype in maptype:
            if each_maptype == 'satellite':
                each_positive_db_map = load_dbimage(datapath=self.database_infos[best_positive_index]['db_satellite_path'], split=self.split, args=self.args)
                each_negative_db_map = [load_dbimage(datapath=self.database_infos[e]['db_satellite_path'], split=self.split, args=self.args) for e in neg_indexes]
            elif each_maptype == 'roadmap':
                each_positive_db_map = load_dbimage(datapath=self.database_infos[best_positive_index]['db_roadmap_path'], split=self.split, args=self.args)
                each_negative_db_map = [load_dbimage(datapath=self.database_infos[e]['db_roadmap_path'], split=self.split, args=self.args) for e in neg_indexes]
            each_negative_db_map = torch.stack(each_negative_db_map, 0)  # [nneg,3,h,w]
            positive_db_map.append(each_positive_db_map)
            negative_db_map.append(each_negative_db_map)
        positive_db_map = torch.stack(positive_db_map, 0)  # [nmap,3,h,w]
        negative_db_map = torch.stack(negative_db_map, 1)  # [nneg,nmap,3,h,w]
        positive_db_eastnorth = torch.tensor([self.database_infos[best_positive_index]['east'], self.database_infos[best_positive_index]['north']]).float()
        negative_db_eastnorth = torch.tensor([[self.database_infos[e]['east'], self.database_infos[e]['north']] for e in neg_indexes]).float()



        # negative_db_map = torch.stack(negative_db_map, 0)  # [nneg,3,h,w]
        db_map = torch.cat((positive_db_map.unsqueeze(0), negative_db_map), 0)  # [1+nneg,nmap,3,h,w]
        db_eastnorth = torch.cat((positive_db_eastnorth.unsqueeze(0), negative_db_eastnorth), 0)  # [1+nneg,2]

        

        triplets_local_indexes = torch.empty((0, 3), dtype=torch.int)
        for neg_num in range(len(neg_indexes)):
            triplets_local_indexes = torch.cat((triplets_local_indexes, torch.tensor([0, 1, 2 + neg_num]).reshape(1, 3)))
        assert query_index < self.queries_num   

        output_dict = {
            'query_image': query_image,
            'query_bev': query_bev,
            'query_sph': query_sph,
            'query_pc': query_pc,
            'query_pc_rgb': query_pc_rgb,
            'query_pc_normal': query_pc_normal,
            'query_eastnorth': query_eastnorth,
            'db_map': db_map,
            'db_eastnorth': db_eastnorth,
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
            for images, indexes in _progress(subset_dl, args=args, desc="cache"):
                images = images.to(args.device)
                features = model(images)
                cache[indexes.numpy()] = features.cpu().numpy()
        return cache
    


    # =================== compute cache separate
    @staticmethod
    def compute_cache_sep(args, model, subset_ds, cache_shape, modelq):
        """Compute the cache containing features of images, which is used to
        find best positive and hardest negatives.

        DDP parallelism: when a process group is initialized, the db and q
        dataloaders are sharded via DistributedSampler so each rank only
        infers 1/world_size of the samples. Results are then all_gather'ed
        so every rank ends up with the full cache.
        """
        import contextlib
        import torch.distributed as dist
        from torch.utils.data.distributed import DistributedSampler

        ddp_on = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if ddp_on else 0
        world = dist.get_world_size() if ddp_on else 1

        model = model.eval()
        modelq = modelq.eval()

        if isinstance(subset_ds, Subset):
            parent_ds = subset_ds.dataset
            subset_indices = [int(i) for i in subset_ds.indices]
        else:
            parent_ds = subset_ds
            subset_indices = list(range(len(parent_ds)))
        db_indices = [i for i in subset_indices if i < parent_ds.database_num]
        q_indices = [i for i in subset_indices if i >= parent_ds.database_num]

        subset_ds_db = Subset(parent_ds, db_indices)
        subset_ds_q = Subset(parent_ds, q_indices)

        if ddp_on:
            db_sampler = DistributedSampler(subset_ds_db, shuffle=False, drop_last=False)
            q_sampler = DistributedSampler(subset_ds_q, shuffle=False, drop_last=False)
        else:
            db_sampler = None
            q_sampler = None

        subset_dl_db = DataLoader(dataset=subset_ds_db, num_workers=args.num_workers,
                                 batch_size=args.infer_batch_size,
                                 sampler=db_sampler, shuffle=False,
                                 pin_memory=(args.device == "cuda"),
                                 collate_fn=nuscenes_collate_fn_cache_db)
        subset_dl_q = DataLoader(dataset=subset_ds_q, num_workers=args.num_workers,
                                batch_size=args.infer_batch_size,
                                sampler=q_sampler, shuffle=False,
                                pin_memory=(args.device == "cuda"),
                                collate_fn=nuscenes_collate_fn_cache_q)

        cache = RAMEfficient2DMatrix(cache_shape, dtype=np.float32) # [db+q, c]

        amp_dt = getattr(args, 'amp_dtype', 'none')
        if amp_dt == 'bf16':
            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        elif amp_dt == 'fp16':
            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
        else:
            amp_ctx = contextlib.nullcontext()

        def _run_and_gather(dataloader, mode, m):
            local_idx_chunks, local_feat_chunks = [], []
            for data_dict, indexes in _progress(dataloader, args=args, desc=f"cache/{mode}", disable=(rank != 0)):
                for _k, _v in data_dict.items():
                    if isinstance(_v, torch.Tensor):
                        data_dict[_k] = _v.to(args.device)
                features = m(data_dict, mode=mode)['embedding'].float()
                local_idx_chunks.append(indexes.cpu())
                local_feat_chunks.append(features.cpu())

            if len(local_idx_chunks) > 0:
                local_idx_np = torch.cat(local_idx_chunks, dim=0).numpy()
                local_feat_np = torch.cat(local_feat_chunks, dim=0).numpy()
            else:
                local_idx_np = np.zeros(0, dtype=np.int64)
                local_feat_np = np.zeros((0, cache_shape[1]), dtype=np.float32)

            if ddp_on:
                gathered = [None] * world
                dist.all_gather_object(gathered, (local_idx_np, local_feat_np))
                for idxs_np, feats_np in gathered:
                    if len(idxs_np) > 0:
                        cache[idxs_np] = feats_np
            else:
                if len(local_idx_np) > 0:
                    cache[local_idx_np] = local_feat_np

        with torch.no_grad(), amp_ctx:
            _run_and_gather(subset_dl_db, 'db', model)
            _run_and_gather(subset_dl_q, 'q', modelq)

        return cache
    



    def get_query_features(self, query_index, cache):
        query_features = cache[query_index + self.database_num]
        if query_features is None:
            query_id = self.queries_infos[query_index].get('sampletoken', query_index)
            raise RuntimeError(
                f"For query {query_id} with index {query_index} features have not been computed!\n"
                "There might be some bug with caching"
            )
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
    


    # =================== random

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
        for query_index in _progress(sampled_queries_indexes, args=args, desc="mine"):
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)
            
            # Choose some random database images, from those remove the soft_positives, and then take the first 10 images as neg_indexes
            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = np.random.choice(self.database_num, size=self.negs_num_per_query+len(soft_positives), replace=False)
            neg_indexes = np.setdiff1d(neg_indexes, soft_positives, assume_unique=True)[:self.negs_num_per_query]
            
            self.triplets_global_indexes.append((query_index, best_positive_index, *neg_indexes))
        # self.triplets_global_indexes is a tensor of shape [1000, 12]
        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes)
    


    # =================== full

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
        for query_index in _progress(sampled_queries_indexes, args=args, desc="mine"):
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
        for query_index in _progress(sampled_queries_indexes, args=args, desc="mine"):
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
        import torch.distributed as dist
        ddp_on = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if ddp_on else 0

        self.triplets_global_indexes = []

        if rank == 0:
            if self.mining in ["partial", 'partial_sep']:
                sampled_queries_indexes = np.random.choice(self.queries_num, args.cache_refresh_rate, replace=False)
            elif self.mining == "msls_weighted":
                sampled_queries_indexes = np.random.choice(self.queries_num, args.cache_refresh_rate, replace=False, p=self.weights)
            sampled_database_indexes = np.random.choice(self.database_num, self.neg_samples_num, replace=False)
            sampled_queries_indexes = sampled_queries_indexes.astype(np.int64)
            sampled_database_indexes = sampled_database_indexes.astype(np.int64)
        else:
            sampled_queries_indexes = np.zeros(args.cache_refresh_rate, dtype=np.int64)
            sampled_database_indexes = np.zeros(self.neg_samples_num, dtype=np.int64)

        if ddp_on:
            device = args.device
            sq_t = torch.from_numpy(sampled_queries_indexes).to(device)
            sd_t = torch.from_numpy(sampled_database_indexes).to(device)
            dist.broadcast(sq_t, src=0)
            dist.broadcast(sd_t, src=0)
            sampled_queries_indexes = sq_t.cpu().numpy()
            sampled_database_indexes = sd_t.cpu().numpy()

        positives_indexes = [self.hard_positives_per_query[i] for i in sampled_queries_indexes]
        positives_indexes = [p for pos in positives_indexes for p in pos]
        database_indexes = list(sampled_database_indexes) + positives_indexes
        database_indexes = list(np.unique(database_indexes))

        subset_ds = Subset(self, list(database_indexes) + list(sampled_queries_indexes + self.database_num))
        cache_length = len(self)
        cache = self.compute_cache_sep(args, model, subset_ds, cache_shape=(cache_length, args.features_dim), modelq=modelq)

        if rank == 0:
            for query_index in _progress(sampled_queries_indexes, args=args, desc="mine"):
                query_features = self.get_query_features(query_index, cache)
                best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)
                soft_positives = self.soft_positives_per_query[query_index]
                neg_indexes = np.setdiff1d(sampled_database_indexes, soft_positives, assume_unique=True)
                neg_indexes = self.get_hardest_negatives_indexes(args, cache, query_features, neg_indexes)
                self.triplets_global_indexes.append((query_index, best_positive_index, *neg_indexes))
            self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes)
        else:
            self.triplets_global_indexes = torch.zeros((0, 0), dtype=torch.long)









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
