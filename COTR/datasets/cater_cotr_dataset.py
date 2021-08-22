'''
COTR dataset
'''

import random
import time

import numpy as np
import torch
from torchvision.transforms import functional as tvtf
from torch.utils import data
import cv2
import imutils

from COTR.datasets import tracking_datasets
from COTR.utils import debug_utils, utils, constants

import COTR.datasets.utils as tracking_utils

from torchvision.transforms import ColorJitter
from PIL import Image

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0)) # H*W*C
    return img

def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def crop(img, center, scale, res, rot=0):
    img = im_to_numpy(img)
  
    # Preprocessing for efficient cropping
    ht, wd = img.shape[0], img.shape[1]
    sf = scale * 200.0 / res[0]
    if sf < 2:
        sf = 1
    else:
        new_size = int(np.math.floor(max(ht, wd) / sf))
        new_ht = int(np.math.floor(ht / sf))
        new_wd = int(np.math.floor(wd / sf))
        if new_size < 2:
            return torch.zeros(res[0], res[1], img.shape[2]) \
                        if len(img.shape) > 2 else torch.zeros(res[0], res[1])
        else:
             #img = imresize(img, [new_ht, new_wd])
             img= cv2.resize(img, dsize=(new_wd,new_ht), interpolation=cv2.INTER_LINEAR)
             center = center * 1.0 / sf
             scale = scale / sf

    # Upper left point
    ul = np.array(transform(np.array([0, 0]), center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(np.array(res), center, scale, res, invert=1))
   
    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
   
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = [max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]]
    new_y = [max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]]
    # Range to sample from original image
    
    
    old_x = [max(0, ul[0]), min(img.shape[1], br[0])]
    old_y = [max(0, ul[1]), min(img.shape[0], br[1])]
    
    
    if(new_x[1]<new_x[0]):
        tmp= new_x[1]
        new_x[1]=new_x[0]
        new_x[0]=tmp
        tmp=old_x[0]
        old_x[0]=old_x[1]
        old_x[1]=tmp
        # swapping the upper-left and bottom right point if upper-left goes out of image
        


    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        #new_img = imrotate(new_img, rot)
        #new_img= imutils.rotate_bound(new_img,-rot) #For bounded rotation
        new_img = imutils.rotate(new_img,rot) 
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = im_to_torch(cv2.resize(new_img, dsize=(res[0],res[1]), interpolation=cv2.INTER_LINEAR))
        #scipy.misc.imresize(new_img, res)
    return new_img

def get_transform(center, scale, res, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def transform(pt, center, scale, res, invert=0, rot=0):
    # pt: 2xN
    # Transform pixel location to different reference
    # pt = pt.T
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    # new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.stack([pt[0] - 1, pt[1] - 1, np.ones_like(pt[0])], axis=0)
    new_pt = np.dot(t, new_pt)
    # print(new_pt.shape)
    return new_pt[:2].astype(int) + 1


class CATERPointTrajDataset(data.Dataset):
    def __init__(self, opt, dataset_type, S=6, root_dir='/projects/katefgroup/datasets', crop_size=(192, 256), use_augs=False):
        assert dataset_type in ['train', 'val', 'test']

        self.opt = opt
        self.dataset_type = dataset_type

        self.max_rotation = 30 # debug
        self.rotation_chance = 0.5 # debug

        num_kp = opt.num_kp
        self.num_kp = num_kp
        self.num_kp2 = num_kp * 2
        self.vis_count = 0

        self.bidirectional = opt.bidirectional
        self.need_rotation = opt.need_rotation
        self.max_rotation = 30
        self.rotation_chance = 0.5

        self.S = S

        self.dataset = tracking_datasets.get_dataset('cater', seqlen=S, shuffle=True, env=dataset_type)

        self.use_augs = use_augs

        # photometric augmentation
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2

        # occlusion augmentation
        self.eraser_aug_prob = 0.5
        self.eraser_bounds = [10, 100]

        # spatial augmentations
        self.crop_size = crop_size
        self.min_scale = -0.1 # 2^this
        self.max_scale = 1.0 # 2^this
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2
        self.do_flip = True
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

    def _trim_corrs(self, in_corrs):
        length = in_corrs.shape[0]
        if length >= self.num_kp:
            mask = np.random.choice(length, self.num_kp)
            return in_corrs[mask]
        else:
            mask = np.random.choice(length, self.num_kp - length)
            return np.concatenate([in_corrs, in_corrs[mask]], axis=0)

    def __len__(self):
        return len(self.dataset)

    def augment_with_rotation(self, query_cap, nn_cap):
        if random.random() < self.rotation_chance:
            theta = np.random.uniform(low=-1, high=1) * self.max_rotation
            query_cap = capture.rotate_capture(query_cap, theta)
        if random.random() < self.rotation_chance:
            theta = np.random.uniform(low=-1, high=1) * self.max_rotation
            nn_cap = capture.rotate_capture(nn_cap, theta)
        return query_cap, nn_cap

    def __getitem__(self, index):
        sample = self.dataset[index]

        # randomly select a query frame and a nn frame 
        pix_T_camXs = sample['pix_T_camXs'].cpu() # S x 4 x 4
        rgb_camXs = sample['rgb_camXs'].cpu() # S x 3 x H x W
        xyz_camXs = sample['xyz_camXs'].cpu() # S x N x 3
        origin_T_camXs = sample['world_T_camXs'].cpu() # S x 4 x 4
        scorelist = sample['scorelist_s'].cpu() # S x K
        lrtlist_camXs = sample['lrtlist_camXs'].cpu() # S x K x 19
        S, _, H, W = rgb_camXs.shape
        _, K = scorelist.shape

        all_keypoints_xy = []
        all_visibles = []

        rgb_camXs = tracking_utils.geom.back2color(rgb_camXs)

        xyz_for_plane = xyz_camXs[:1].clone()
        xyz_for_plane[:,:,2] = xyz_for_plane[:,:,2].clamp(0,20)
        fg_inds = torch.logical_not(tracking_utils.geom.get_plane_inds(xyz_for_plane)).reshape(-1)

        depth_camXs, _ = tracking_utils.geom.create_depth_image(pix_T_camXs, xyz_camXs, H, W)

        for obj_id in range(K+1):
            nn_frame_id = 0

            # Get xyzs to track
            if obj_id == K:
                # background
                nn_frame_id = 0
                inb = None
                for ooid in range(10):
                    if scorelist[nn_frame_id, ooid] == 0:
                        continue
                    if inb is None:
                        inb = tracking_utils.geom.get_pts_inbound_lrt(xyz_camXs[nn_frame_id:nn_frame_id+1], lrtlist_camXs[nn_frame_id:nn_frame_id+1, ooid], add_pad=0.1).reshape(-1) # N
                    else:
                        inb += tracking_utils.geom.get_pts_inbound_lrt(xyz_camXs[nn_frame_id:nn_frame_id+1], lrtlist_camXs[nn_frame_id:nn_frame_id+1, ooid], add_pad=0.1).reshape(-1) # N
                inb = (inb == 0)
            else:
                # object
                if scorelist[0, obj_id] == 0:
                    continue

                inb = fg_inds * tracking_utils.geom.get_pts_inbound_lrt(xyz_camXs[nn_frame_id:nn_frame_id+1], lrtlist_camXs[nn_frame_id:nn_frame_id+1, obj_id], add_pad=-0.03).reshape(-1) # N
                if inb.sum() < 50:
                    continue
            
            nn_xyz_camXs = xyz_camXs[nn_frame_id:nn_frame_id+1, inb] # 1, N, 3
            nn_xy_camXs = tracking_utils.geom.camera2pixels(nn_xyz_camXs, pix_T_camXs[nn_frame_id:nn_frame_id+1])
            nn_keypoints_xy = nn_xy_camXs[0].cpu().numpy() # N x 2

            nn_keypoints_x = nn_xy_camXs[..., 0] # 1 x N
            nn_keypoints_y = nn_xy_camXs[..., 1] # 1 x N, in image coord

            keypoints_xy = []
            visibles = []
            for query_frame_id in range(1, S):
                '''
                # camera transformation
                origin_T_nn = origin_T_camXs[nn_frame_id:nn_frame_id+1]
                origin_T_query = origin_T_camXs[query_frame_id:query_frame_id+1]
                query_T_nn_cam = torch.matmul(tracking_utils.geom.safe_inverse(origin_T_query), origin_T_nn)
                '''
                query_T_nn_cam = tracking_utils.geom.eye_4x4(1, device='cpu').squeeze(1)

                # lrt (object) transformation
                if obj_id != K:
                    mat1 = lrtlist_camXs[nn_frame_id, obj_id][3:].reshape(4,4)
                    mat2 = lrtlist_camXs[query_frame_id, obj_id][3:].reshape(4,4)
                    query_T_nn_obj = torch.matmul(mat2, mat1.inverse()).unsqueeze(0)

                    # compose transforms
                    query_T_nn = torch.matmul(query_T_nn_obj, query_T_nn_cam)
                else:
                    query_T_nn = query_T_nn_cam

                # query xyz
                query_xyz_camXs = tracking_utils.geom.apply_4x4(query_T_nn, nn_xyz_camXs)
                query_xy_camXs = tracking_utils.geom.camera2pixels(query_xyz_camXs, pix_T_camXs[query_frame_id:query_frame_id+1]) # 
                query_keypoints_xy = query_xy_camXs[0].cpu().numpy() # N x 2

                # obtain visible / non-visible
                query_d = depth_camXs[query_frame_id, 0, query_xy_camXs[0,:,1].round().long().clamp(0, H-1), query_xy_camXs[0,:,0].round().long().clamp(0, W-1)]
                depth_dist = (query_d - query_xyz_camXs[:,:,2].reshape(-1)).abs()
                visible = (depth_dist < 0.05).reshape(-1).cpu().numpy() # N

                if query_frame_id == 1:
                    keypoints_xy.append(nn_keypoints_xy)
                    keypoints_xy.append(np.copy(query_keypoints_xy))
                    visibles.append(np.copy(visible))
                else:
                    keypoints_xy.append(np.copy(query_keypoints_xy))
                    visibles.append(np.copy(visible))

            keypoints_xy = np.stack(keypoints_xy, 0)
            visibles = np.stack(visibles, 0)
            mask = np.logical_and(((keypoints_xy[:,:,0] > 0) * (keypoints_xy[:,:,0] < W)).sum(0) == self.S, ((keypoints_xy[:,:,1] > 0) * (keypoints_xy[:,:,1] < H)).sum(0) == self.S)
            keypoints_xy = keypoints_xy[:,mask]
            visibles = visibles[:,mask]

            all_keypoints_xy.append(keypoints_xy)
            all_visibles.append(visibles)

        for i in range(len(all_keypoints_xy)):
            mask = np.random.choice(all_keypoints_xy[i].shape[1], self.num_kp2 // 2)
            all_keypoints_xy[i] = all_keypoints_xy[i][:, mask]
            all_visibles[i] = all_visibles[i][:, mask]

        if len(all_keypoints_xy) > 1:
            fg_keypoints_xy = np.concatenate(all_keypoints_xy[:-1], 1)
            fg_visibles = np.concatenate(all_visibles[:-1], 1)
            mask = np.random.choice(self.num_kp2 // 2 * (len(all_keypoints_xy)-1), self.num_kp2 // 2)
            fg_keypoints_xy = fg_keypoints_xy[:, mask]
            fg_visibles = fg_visibles[:, mask]

            bg_keypoints_xy = all_keypoints_xy[-1]
            bg_visibles = all_visibles[-1]

            keypoints_xy = np.concatenate([fg_keypoints_xy, bg_keypoints_xy], 1)
            visibles = np.concatenate([fg_visibles, bg_visibles], 1)
            visibles = np.concatenate([torch.ones(self.num_kp2).reshape(1,-1), visibles], 0)
        else:
            keypoints_xy = all_keypoints_xy[-1]
            visibles = all_visibles[-1]
            visibles = np.concatenate([torch.ones(self.num_kp2//2).reshape(1,-1), visibles], 0)

        kp_xy = torch.from_numpy(keypoints_xy).unsqueeze(0)
        vis = torch.from_numpy(visibles).unsqueeze(0)
        kp_xy = torch.from_numpy(keypoints_xy).unsqueeze(0)
        inds = tracking_utils.geom.farthest_point_sample(kp_xy[:,0], self.num_kp2)
        kp_xy = kp_xy[0,:,inds[0]]
        vis = vis[0,:,inds[0]]
        keypoints_xy = kp_xy.detach().cpu().numpy()
        visibles = vis.detach().cpu().numpy()

        if self.use_augs:
            rgbs, keypoints_xy, visibles = self.augment(rgb_camXs.numpy(), keypoints_xy.transpose(1,0,2), visibles.transpose(1,0))

            rgbs = torch.from_numpy(rgbs.transpose(0,3,1,2).copy()).numpy()
            keypoints_xy = keypoints_xy.transpose(1,0,2)
            visibles = visibles.transpose(1,0)
        else:
            rgbs = rgb_camXs.numpy()

        S, C, H, W = rgbs.shape 

        rand_frame_id = np.random.choice(np.arange(S), 2, replace=False)
        query_frame_id = rand_frame_id[0]
        nn_frame_id = rand_frame_id[1]

        query_img = rgbs[query_frame_id] / 255 # 3, H, W
        nn_img = rgbs[nn_frame_id] / 255 # 3, H, W
        query_keypoints_xy = keypoints_xy[query_frame_id]
        nn_keypoints_xy = keypoints_xy[nn_frame_id]

        if random.random() < self.rotation_chance:
            theta1 = np.random.uniform(low=-1, high=1) * self.max_rotation
        else:
            theta1 = 0.0

        if random.random() < self.rotation_chance:
            theta2 = np.random.uniform(low=-1, high=1) * self.max_rotation
        else:
            theta2 = 0.0

        c1 = [W/2, H/2]
        s1 = np.random.uniform(low=0.6, high=1.0)

        c2 = [W/2, H/2]
        s2 = np.random.uniform(low=0.6, high=1.0)

        c1[0] += np.random.uniform(low=-1, high=1) * 50
        c2[1] += np.random.uniform(low=-1, high=1) * 50

        c1[0] += np.random.uniform(low=-1, high=1) * 50
        c2[1] += np.random.uniform(low=-1, high=1) * 50

        query_img = crop(query_img, c1, s1, (constants.MAX_SIZE, constants.MAX_SIZE), rot=theta1)
        query_keypoints_xy = transform(query_keypoints_xy.T, c1, s1, (constants.MAX_SIZE, constants.MAX_SIZE), rot=theta1).T

        nn_img = crop(nn_img, c2, s2, (constants.MAX_SIZE, constants.MAX_SIZE), rot=theta2)
        nn_keypoints_xy = transform(nn_keypoints_xy.T, c2, s2, (constants.MAX_SIZE, constants.MAX_SIZE), rot=theta2).T

        H, W = constants.MAX_SIZE, constants.MAX_SIZE

        sbs_img = torch.cat([query_img, nn_img], axis=-1) # 3 x H x 2*W

        corrs = np.concatenate([query_keypoints_xy, nn_keypoints_xy], axis=1) # N x 4, x1y1x2y2
        mask_query = np.logical_and(np.logical_and(query_keypoints_xy[:,0]>0, query_keypoints_xy[:,0]<W), np.logical_and(query_keypoints_xy[:,1]>0, query_keypoints_xy[:,1]<H))
        mask_nn = np.logical_and(np.logical_and(nn_keypoints_xy[:,0]>0, nn_keypoints_xy[:,0]<W), np.logical_and(nn_keypoints_xy[:,1]>0, nn_keypoints_xy[:,1]<H))
        mask = np.logical_and(mask_nn, mask_query)
        corrs = corrs[mask]

        if len(corrs) == 0:
            # print('bad example')
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        corrs = self._trim_corrs(corrs)

        corrs[:, 2] += W
        corrs = corrs.astype(float)
        corrs /= np.array([W * 2, H, W * 2, H]).reshape(1, 4).astype(float)
        assert (0.0 <= corrs[:, 0]).all() and (corrs[:, 0] <= 0.5).all()
        assert (0.0 <= corrs[:, 1]).all() and (corrs[:, 1] <= 1.0).all()
        assert (0.5 <= corrs[:, 2]).all() and (corrs[:, 2] <= 1.0).all()
        assert (0.0 <= corrs[:, 3]).all() and (corrs[:, 3] <= 1.0).all()
        out = {
            'image': tvtf.normalize(sbs_img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            'corrs': torch.from_numpy(corrs).float(),
        }

        if self.bidirectional:
            out['queries'] = torch.from_numpy(np.concatenate([corrs[:, :2], corrs[:, 2:]], axis=0)).float()
            out['targets'] = torch.from_numpy(np.concatenate([corrs[:, 2:], corrs[:, :2]], axis=0)).float()
        else:
            out['queries'] = torch.from_numpy(corrs[:, :2]).float()
            out['targets'] = torch.from_numpy(corrs[:, 2:]).float()

        return out

    def augment(self, rgbs, trajs, visibles):
        '''
        Input:
            rgbs --- np.array(S, C, H, W)
                     list of len S, each = np.array (H, W, 3)
            trajs --- np.array (N, T, 2)
            visibles --- np.array(N, T)
        Output:
            rgbs_aug --- np.array (S, H, W, 3)
            trajs_aug --- np.array (N_new, T, 2)
            visibles_aug --- np.array (N_new, T)
        '''
        # rgbs = [((rgbs[i].transpose((1,2,0))+0.5)*255).astype(np.uint8) for i in range(rgbs.shape[0])]
        rgbs = [rgbs[i].transpose((1,2,0)).astype(np.uint8) for i in range(rgbs.shape[0])]
        
        S = len(rgbs)
        H, W = rgbs[0].shape[:2]

        ############ photometric augmentation ############
        if np.random.rand() < self.asymmetric_color_aug_prob:
            for i in range(S):
                rgbs[i] = np.array(self.photo_aug(Image.fromarray(rgbs[i])), dtype=np.uint8)
        else:
            image_stack = np.concatenate(rgbs, axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            rgbs = np.split(image_stack, S, axis=0)

        ############ eraser transform (per image) ############
        for i in range(1, S):
            if np.random.rand() < self.eraser_aug_prob:
                mean_color = np.mean(rgbs[i].reshape(-1, 3), axis=0)
                for _ in range(np.random.randint(1, 6)):
                    xc = np.random.randint(0, W)
                    yc = np.random.randint(0, H)
                    dx = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                    dy = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                    x0 = np.clip(xc - dx/2, 0, W-1).round().astype(np.int32)
                    x1 = np.clip(xc + dx/2, 0, W-1).round().astype(np.int32)
                    y0 = np.clip(yc - dy/2, 0, W-1).round().astype(np.int32)
                    y1 = np.clip(yc + dy/2, 0, W-1).round().astype(np.int32)
                    # print(x0, x1, y0, y1)
                    rgbs[i][y0:y1, x0:x1, :] = mean_color

                    occ_inds = np.logical_and(np.logical_and(trajs[:,i,0] >= x0, trajs[:,i,0] < x1), np.logical_and(trajs[:,i,1] >= y0, trajs[:,i,1] < y1))
                    visibles[occ_inds,i] = 0

        ############ spatial transform ############
        image_stack = np.concatenate(rgbs, axis=0)
        # scaling + stretching
        scale_x = 1.0
        scale_y = 1.0
        H_new = H
        W_new = W
        if np.random.rand() < self.spatial_aug_prob:
            min_scale = np.maximum(
                (self.crop_size[0] + 8) / float(H),
                (self.crop_size[1] + 8) / float(W))

            scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
            scale_x = scale 
            scale_y = scale 

            if np.random.rand() < self.stretch_prob:
                scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
                scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

            scale_x = np.clip(scale_x, min_scale, None)
            scale_y = np.clip(scale_y, min_scale, None)

            H_new = int(H * scale_y)
            W_new = int(W * scale_x)
            dim_resize = (W_new, H_new * S)
            image_stack = cv2.resize(image_stack, dim_resize, interpolation=cv2.INTER_LINEAR)

        # flip
        h_flipped = False
        v_flipped = False
        if self.do_flip:
            # h flip
            if np.random.rand() < self.h_flip_prob:
                h_flipped = True
                image_stack = image_stack[:, ::-1]

            # v flip
            if np.random.rand() < self.v_flip_prob:
                v_flipped = True
                image_stack = image_stack[::-1, :]

        y0 = np.random.randint(0, H_new - self.crop_size[0])
        x0 = np.random.randint(0, W_new - self.crop_size[1])

        rgbs = np.stack(np.split(image_stack, S, axis=0), axis=0)
        rgbs = rgbs[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        if v_flipped:
            rgbs = rgbs[::-1]
        
        # transform trajs
        trajs[:, :, 0] *= scale_x
        trajs[:, :, 1] *= scale_y
        if h_flipped:
            trajs[:, :, 0] = W_new - trajs[:, :, 0]
        if v_flipped:
            trajs[:, :, 1] = H_new - trajs[:, :, 1]
        trajs[:, :, 0] -= x0
        trajs[:, :, 1] -= y0

        inbound = (trajs[:, 0, 0] >= 0) & (trajs[:, 0, 0] < self.crop_size[1]) & (trajs[:, 0, 1] >= 0) & (trajs[:, 0, 1] < self.crop_size[0])
        trajs = trajs[inbound]
        visibles = visibles[inbound]

        return rgbs, trajs, visibles


