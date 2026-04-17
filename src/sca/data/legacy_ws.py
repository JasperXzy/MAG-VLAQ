
import os
import torch
import logging
import numpy as np
from glob import glob
from PIL import Image
from os.path import join
import torch.utils.data as data
import torchvision.transforms as T
from sklearn.neighbors import NearestNeighbors
import random
from torchvision.transforms import InterpolationMode
import torchvision.transforms as TVT
from sca.models.layers.sparse_utils import batched_coordinates, sparse_quantize
import copy
import matplotlib.pyplot as plt

from sca.data.pc_augmentation import (
    PCRandomFlip,
    PCRandomRotation,
    PCRandomTranslation,
    PCRandomScale,
    PCRandomShear,
    PCJitterPoints,
    PCRemoveRandomPoints,
    PCRemoveRandomBlock
)


USE_MM_QUERY = True


base_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def path_to_pil_img(path):
    image = Image.open(path)
    image = image.convert("RGB")
    return image




def collate_fn(batch):
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
    images = torch.cat([e[0]['image'] for e in batch]) # [12,c,h,w] -> [b*12,c,h,w]
    bevs = torch.stack([e[0]['bev'] for e in batch]) # [3,h,w] -> [b,3,h,w]
    sphs = torch.stack([e[0]['sph'] for e in batch]) # [3,h,w] -> [b,3,h,w]
    pcs = [e[0]['pc'] for e in batch]
    # ---- batch augmentation
    if USE_MM_QUERY:
        coords = torch.ones(1)
    else:
        coords = batched_coordinates(pcs)
        batchids = coords[:,:1]
        coords = coords[:,1:]
        coords = PCRandomRotation(max_theta=5, max_theta2=0, axis=np.array([0, 0, 1]))(coords) # CPU intense
        coords = torch.cat([batchids, coords], dim=1)

    feats = torch.ones([coords.shape[0], 1]).float()
    triplets_local_indexes = torch.cat([e[1][None] for e in batch])
    triplets_global_indexes = torch.cat([e[2][None] for e in batch])
    for i, (local_indexes, global_indexes) in enumerate(zip(triplets_local_indexes, triplets_global_indexes)):
        local_indexes += len(global_indexes) * i  # Increment local indexes by offset (len(global_indexes) is 12)
    output_dict = {
        'image': images,
        'bev': bevs,
        'sph': sphs,
        'coord': coords,
        'feat': feats,
    }
    # return images, torch.cat(tuple(triplets_local_indexes)), triplets_global_indexes
    return output_dict, torch.cat(tuple(triplets_local_indexes)), triplets_global_indexes





def collate_fn_cache_db(batch):
    images = torch.stack([e[0]['image'] for e in batch]) # [12,c,h,w] -> [b*12,c,h,w]
    bevs = torch.empty([images.shape[0], 0])
    sphs = torch.empty([images.shape[0], 0])
    coords = torch.empty([images.shape[0], 0])
    feats = torch.empty([images.shape[0], 0])
    indices = torch.tensor([e[1] for e in batch])
    output_dict = {
        'image': images,
        'bev': bevs,
        'sph': sphs,
        'coord': coords,
        'feat': feats,
    }
    return output_dict, indices




def collate_fn_cache_q(batch):
    '''
    output of collate_fn should be applicable with .to(device)
    '''
    images = torch.stack([e[0]['image'] for e in batch]) # [12,c,h,w] -> [b*12,c,h,w]
    bevs = torch.stack([e[0]['bev'] for e in batch]) # [3,h,w] -> [b,3,h,w]
    sphs = torch.stack([e[0]['sph'] for e in batch]) # [3,h,w] -> [b,3,h,w]
    pcs = [e[0]['pc'] for e in batch]
    # ---- batch augmentation
    if USE_MM_QUERY:
        coords = torch.ones(1)
    else:
        coords = batched_coordinates(pcs)
        batchids = coords[:,:1]
        coords = coords[:,1:]
        coords = PCRandomRotation(max_theta=5, max_theta2=0, axis=np.array([0, 0, 1]))(coords) # CPU intense
        coords = torch.cat([batchids, coords], dim=1)

    feats = torch.ones([coords.shape[0], 1]).float()
    indices = torch.tensor([e[1] for e in batch])
    output_dict = {
        'image': images,
        'bev': bevs,
        'sph': sphs,
        'coord': coords,
        'feat': feats
    }
    return output_dict, indices





def generate_bev_from_pc(pc, w=200, max_thd=2):
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
    # kitti   361  61
    # ithaca  361  101
    if args is not None and 'ithaca365' in args.dataset_name:
        w = 361
        h = 101
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
    




def load_bev(file_path, dataset_name, bev_w, args): # filename is the same as load_pc
    """load pc + bev + sph
    """
    # # 1. load bev
    # filename = filename.split('/')
    # filename[-1] = filename[-1].replace('bin', 'png') 
    # filename[-2] = filename[-2] + f'_w{args.bev_w}_{args.bev_cmap}'
    # filename = os.path.join(*filename)
    # file_path = os.path.join(self.data_root_dir, filename)
    # assert os.path.exists(file_path)
    # bev = Image.open(file_path)

    # 2. genrated bev from pc
    # file_path = os.path.join(self.data_root_dir, filename)
    if 'kitti_raw' in dataset_name:
        pc = np.fromfile(file_path, dtype=np.float32).reshape(-1,4)[:,:3] # kitti 
        assert pc.shape[1] == 3  
        bev = generate_bev_from_pc(pc, w=bev_w-1, max_thd=100)
        sph = generate_sph_from_pc(pc, args=args)
    elif 'ithaca365' in dataset_name:
        pc = np.load(file_path)
        assert pc.shape[1] == 3
        bev = generate_bev_from_pc(pc, w=bev_w-1, max_thd=100)
        sph = generate_sph_from_pc(pc, args=args)
    else:
        raise NotImplementedError
    
    # ==== pc
    if USE_MM_QUERY:
        pc = torch.tensor(pc).float()
    else:
        pc = torch.tensor(pc).float()
        tfpc = TVT.Compose([
            PCJitterPoints(sigma=0.001, clip=0.002), 
            PCRemoveRandomPoints(r=(0.0, 0.1)),
            PCRandomTranslation(max_delta=0.01), 
            PCRemoveRandomBlock(p=0.4),
            # PCRandomRotation(max_theta=5, max_theta2=0, axis=np.array([0, 0, 1])), # CPU intense
            PCRandomFlip([0.25, 0.25, 0.]),
        ])
        pc = tfpc(pc)
        pc = sparse_quantize(coordinates=pc, quantization_size=args.quant_size)



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
    
    tf = TVT.Compose([
        TVT.Resize(resize_size, interpolation=resize_mode),
        TVT.RandomRotation(args.bev_rotate, interpolation=rotate_mode),
        TVT.CenterCrop(args.bev_cropsize),
        TVT.ColorJitter(brightness=args.bev_jitter, contrast=args.bev_jitter, saturation=args.bev_jitter, hue=min(0.5, args.bev_jitter)),
        TVT.ToTensor(),
        TVT.Normalize(mean=args.bev_mean, std=args.bev_std)
    ])
    bev = tf(bev)

    # # ---- DEBUG:viz
    # bev = bev*args.bev_std + args.bev_mean
    # bev = TVT.ToPILImage()(bev)
    # bev.show()



    # ==== sph
    sph = Image.fromarray(sph).convert('RGB')
    (_w,_h) = sph.size
    assert args.sph_resize <= 1
    resize_ratio = random.uniform(args.sph_resize, 2-args.sph_resize)
    resize_size = int(resize_ratio*min(_w,_h))
    tf = TVT.Compose([
        TVT.Resize(resize_size, interpolation=InterpolationMode.NEAREST),
        TVT.ColorJitter(brightness=args.sph_jitter, contrast=args.sph_jitter, saturation=args.sph_jitter, hue=min(0.5, args.sph_jitter)),
        TVT.ToTensor(),
        TVT.Normalize(mean=args.sph_mean, std=args.sph_std)
    ])
    sph = tf(sph)


    return bev, pc, sph












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









# =================== test dataset

class BaseDataset(data.Dataset):
    """Dataset with images from database and queries, used for inference (testing and building cache).
    """
    def __init__(self, args, datasets_folder="datasets", dataset_name="pitts30k", split="train"):
        super().__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.dataset_folder = join(datasets_folder, dataset_name, "images", split)
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
        
        self.resize = args.resize
        self.test_method = args.test_method
        
        #### Read paths and UTM coordinates for all images.
        database_folder = join(self.dataset_folder, "database")
        queries_folder = join(self.dataset_folder, "queries")
        if not os.path.exists(database_folder):
            raise FileNotFoundError(f"Folder {database_folder} does not exist")
        if not os.path.exists(queries_folder):
            raise FileNotFoundError(f"Folder {queries_folder} does not exist")
        if 'kitti' in dataset_name:
            self.database_paths = sorted(glob(join(database_folder, "**", "*.png"), recursive=True))
            self.queries_paths = sorted(glob(join(queries_folder, "**", "*.png"),  recursive=True))
            self.queries_pc_paths = [e.replace('/queries/','/queries_pc/').replace('.png','.bin') for e in self.queries_paths]
        elif 'ithaca365' in dataset_name:
            self.database_paths = sorted(glob(join(database_folder, "**", "*.png"), recursive=True))
            self.queries_paths = sorted(glob(join(queries_folder, "**", "*.png"),  recursive=True))
            self.queries_pc_paths = sorted(glob(join(queries_folder.replace('/queries','/queries_pc'), "**", "*.npy"),  recursive=True))
        else:
            self.database_paths = sorted(glob(join(database_folder, "**", "*.jpg"), recursive=True))
            self.queries_paths = sorted(glob(join(queries_folder, "**", "*.jpg"),  recursive=True))
            self.queries_pc_paths = [e.replace('/queries/','/queries_pc/').replace('.png','.bin') for e in self.queries_paths]
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]).astype(float)
        self.queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]).astype(float)
        
        # Find soft_positives_per_query, which are within val_positive_dist_threshold (deafult 25 meters)
        # knn = NearestNeighbors(n_jobs=-1)
        knn = NearestNeighbors(n_jobs=self.args.num_workers + 1)
        knn.fit(self.database_utms)
        self.soft_positives_per_query = knn.radius_neighbors(self.queries_utms,
                                                             radius=args.val_positive_dist_threshold,
                                                             return_distance=False)
        
        self.images_paths = list(self.database_paths) + list(self.queries_paths)
        self.pcs_paths = ['']*len(self.database_paths) + list(self.queries_pc_paths)
        
        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)
    
    def __getitem__(self, index):
        img = path_to_pil_img(self.images_paths[index])
        img = base_transform(img)
        # With database images self.test_method should always be "hard_resize"
        if self.test_method == "hard_resize":
            # self.test_method=="hard_resize" is the default, resizes all images to the same size.
            img = T.functional.resize(img, self.resize, antialias=True)
        else:
            img = self._test_query_transform(img)
        if index >= self.database_num: # query
            bev, pc, sph = load_bev(
                self.pcs_paths[index],
                dataset_name=self.dataset_name,
                bev_w=self.args.bev_w,
                args=self.args,
            )
        else: # database
            bev = torch.empty(0)
            sph = torch.empty(0)
            pc = torch.empty(0)
        output_dict = {
            'image': img,
            'bev': bev,
            'sph': sph,
            'sph': sph,
            'pc': pc,
        }
        # return img, index
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
        return len(self.images_paths)
    
    def __repr__(self):
        return f"< {self.__class__.__name__}, {self.dataset_name} - #database: {self.database_num}; #queries: {self.queries_num} >"
    
    def get_positives(self):
        return self.soft_positives_per_query















# =================== train dataset

class TripletsDataset(BaseDataset):
    """Dataset used for training from precomputed triplet indexes."""
    def __init__(self, args, datasets_folder="datasets", dataset_name="pitts30k", split="train", negs_num_per_query=10):
        super().__init__(args, datasets_folder, dataset_name, split)
        self.mining = args.mining
        self.neg_samples_num = args.neg_samples_num  # Number of negatives to randomly sample
        self.negs_num_per_query = negs_num_per_query  # Number of negatives per query in each batch
        if self.mining == "full":  # "Full database mining" keeps a cache with last used negatives
            self.neg_cache = [np.empty((0,), dtype=np.int32) for _ in range(self.queries_num)]
        self.is_inference = False
        
        identity_transform = T.Lambda(lambda x: x)
        self.resized_transform = T.Compose([
            T.Resize(self.resize, antialias=True) if self.resize is not None else identity_transform,
            T.ColorJitter(args.color_jitter, args.color_jitter, args.color_jitter, min(args.color_jitter,0.5)),
            base_transform
        ])
        
        self.query_transform = T.Compose([
                # T.ColorJitter(args.brightness, args.contrast, args.saturation, args.hue),
                # T.ColorJitter(args.color_jitter, args.color_jitter, args.color_jitter, min(args.color_jitter,0.5)),
                # T.RandomPerspective(args.rand_perspective),
                # T.RandomResizedCrop(size=self.resize, scale=(1-args.random_resized_crop, 1), antialias=True),
                # T.RandomRotation(degrees=args.random_rotation),
                self.resized_transform,
                # T.Resize(args.resize, antialias=True)
        ])
        
        # Find hard_positives_per_query, which are within train_positives_dist_threshold (10 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.hard_positives_per_query = list(knn.radius_neighbors(self.queries_utms,
                                             radius=args.train_positives_dist_threshold,  # 10 meters
                                             return_distance=False))
        
        #### Some queries might have no positive, we should remove those queries.
        queries_without_any_hard_positive = np.where(np.array([len(p) for p in self.hard_positives_per_query], dtype=object) == 0)[0]
        if len(queries_without_any_hard_positive) != 0:
            logging.info(f"There are {len(queries_without_any_hard_positive)} queries without any positives " +
                         "within the training set. They won't be considered as they're useless for training.")
        # Remove queries without positives
        # self.hard_positives_per_query = np.delete(self.hard_positives_per_query, queries_without_any_hard_positive)
        self.hard_positives_per_query = [p for i, p in enumerate(self.hard_positives_per_query) if i not in queries_without_any_hard_positive]

        # self.queries_paths = np.delete(self.queries_paths, queries_without_any_hard_positive)
        self.queries_paths = [p for i, p in enumerate(self.queries_paths) if i not in queries_without_any_hard_positive]
        self.queries_pc_paths = [p for i, p in enumerate(self.queries_pc_paths) if i not in queries_without_any_hard_positive]
        
        # Recompute images_paths and queries_num because some queries might have been removed
        self.images_paths = list(self.database_paths) + list(self.queries_paths)
        self.queries_num = len(self.queries_paths)
        
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
        query = self.query_transform(path_to_pil_img(self.queries_paths[query_index]))
        positive = self.resized_transform(path_to_pil_img(self.database_paths[best_positive_index]))
        negatives = [self.resized_transform(path_to_pil_img(self.database_paths[i])) for i in neg_indexes]
        images = torch.stack((query, positive, *negatives), 0)
        triplets_local_indexes = torch.empty((0, 3), dtype=torch.int)
        for neg_num in range(len(neg_indexes)):
            triplets_local_indexes = torch.cat((triplets_local_indexes, torch.tensor([0, 1, 2 + neg_num]).reshape(1, 3)))
        

        assert query_index < self.queries_num        
        query_bev, query_pc, query_sph = load_bev(
            self.queries_pc_paths[query_index],
            dataset_name=self.dataset_name,
            bev_w=self.args.bev_w,
            args=self.args,
        ) # load + transform

        output_dict = {
            'image':images,
            'bev':query_bev,
            'sph': query_sph,
            'pc':query_pc,
        }
        # return images, triplets_local_indexes, self.triplets_global_indexes[index]
        return output_dict, triplets_local_indexes, self.triplets_global_indexes[index]
    
    def __len__(self):
        if self.is_inference:
            # At inference time return the number of images. This is used for caching or computing NetVLAD's clusters
            length = super().__len__()
            return length
        else:
            length = len(self.triplets_global_indexes)
            return length
