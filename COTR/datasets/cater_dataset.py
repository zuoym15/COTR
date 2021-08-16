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

# from COTR.datasets import megadepth_dataset
from COTR.datasets import tracking_datasets
from COTR.utils import debug_utils, utils, constants
# from COTR.projector import pcd_projector
# from COTR.cameras import capture
# from COTR.utils.utils import CropCamConfig
# from COTR.inference import inference_helper
# from COTR.inference.inference_helper import two_images_side_by_side

import COTR.datasets.utils as tracking_utils

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


class CATERDataset(data.Dataset):
    def __init__(self, opt, dataset_type: str):
        assert dataset_type in ['train', 'val', 'test']
        # assert len(opt.scenes_name_list) > 0
        self.opt = opt
        self.dataset_type = dataset_type
        # self.sfm_dataset = megadepth_dataset.MegadepthDataset(opt, dataset_type)

        self.kp_pool = opt.kp_pool
        self.num_kp = opt.num_kp
        self.bidirectional = opt.bidirectional
        self.need_rotation = opt.need_rotation
        # self.max_rotation = opt.max_rotation
        # self.rotation_chance = opt.rotation_chance
        self.max_rotation = 30 # debug
        self.rotation_chance = 0.5 # debug

        self.dataset = tracking_datasets.get_dataset('cater', seqlen=6, shuffle=True, env=dataset_type)

        self.vis_count = 0
        self.occlusion = {}

    def _trim_corrs(self, in_corrs):
        length = in_corrs.shape[0]
        if length >= self.num_kp:
            mask = np.random.choice(length, self.num_kp)
            return in_corrs[mask]
        else:
            mask = np.random.choice(length, self.num_kp - length)
            return np.concatenate([in_corrs, in_corrs[mask]], axis=0)

    def __len__(self):
        # if self.dataset_type == 'val':
        #     return min(1000, self.sfm_dataset.num_queries)
        # else:
        #     return self.sfm_dataset.num_queries
        return len(self.dataset)

    def augment_with_rotation(self, query_cap, nn_cap):
        if random.random() < self.rotation_chance:
            theta = np.random.uniform(low=-1, high=1) * self.max_rotation
            query_cap = capture.rotate_capture(query_cap, theta)
        if random.random() < self.rotation_chance:
            theta = np.random.uniform(low=-1, high=1) * self.max_rotation
            nn_cap = capture.rotate_capture(nn_cap, theta)
        return query_cap, nn_cap

    def __getitem_2view__(self, index):
        assert self.opt.k_size == 1
        
        sample = self.dataset[index]
        # if self.need_rotation:
        #     query_cap, nn_cap = self.augment_with_rotation(query_cap, nn_cap)

        # randomly select a query frame and a nn frame 
        pix_T_camXs = sample['pix_T_camXs'].cuda() # S x 4 x 4
        rgb_camXs = sample['rgb_camXs'].cuda() # S x 3 x H x W
        xyz_camXs = sample['xyz_camXs'].cuda() # S x N x 3
        origin_T_camXs = sample['world_T_camXs'].cuda() # S x 4 x 4
        scorelist = sample['scorelist_s'].cuda() # S x K
        lrtlist_camXs = sample['lrtlist_camXs'].cuda() # S x K x 19
        S, _, H, W = rgb_camXs.shape
        _, K = scorelist.shape

        rgb_camXs += .5 # range [0,1]

        rand_frame_id = np.random.choice(np.arange(S), 2, replace=False)
        #query_frame_id = rand_frame_id[0]
        #nn_frame_id = rand_frame_id[1]

        nn_frame_id = 0
        query_frame_id = 1

        filtered_xyzs = []
        # only take points belong to objects
        for obj_id in range(10):
            if scorelist[nn_frame_id, obj_id] == 0:
                continue
            inb = tracking_utils.geom.get_pts_inbound_lrt(xyz_camXs[nn_frame_id:nn_frame_id+1], lrtlist_camXs[nn_frame_id:nn_frame_id+1, obj_id], add_pad=0).reshape(-1) # N
            xyz = xyz_camXs[nn_frame_id:nn_frame_id+1, inb] # 1, N, 3
            filtered_xyzs.append(xyz)

        nn_xyz_camXs = torch.cat(filtered_xyzs, dim=1)

        # nn_xyz_camXs = xyz_camXs[nn_frame_id:nn_frame_id+1]

        nn_xy_camXs = tracking_utils.geom.camera2pixels(nn_xyz_camXs, pix_T_camXs[nn_frame_id:nn_frame_id+1]) # 1 x N x 2
        nn_keypoints_x = nn_xy_camXs[..., 0] # 1 x N
        nn_keypoints_y = nn_xy_camXs[..., 1] # 1 x N, in image coord

        # transform the xyzs into query frame, to find correspondence
        origin_T_nn = origin_T_camXs[nn_frame_id:nn_frame_id+1]
        origin_T_query = origin_T_camXs[query_frame_id:query_frame_id+1]

        query_T_nn = torch.matmul(tracking_utils.geom.safe_inverse(origin_T_query), origin_T_nn)
        query_xyz_camXs = tracking_utils.geom.apply_4x4(query_T_nn, nn_xyz_camXs)

        query_xy_camXs = tracking_utils.geom.camera2pixels(query_xyz_camXs, pix_T_camXs[query_frame_id:query_frame_id+1]) # 

        nn_keypoints_xy = nn_xy_camXs[0].cpu().numpy() # N x 2
        query_keypoints_xy = query_xy_camXs[0].cpu().numpy() # N x 2

        # obtain visible / non-visible
        depth, valid = tracking_utils.geom.create_depth_image(pix_T_camXs[query_frame_id:query_frame_id+1], xyz_camXs[query_frame_id:query_frame_id+1], H, W)
        query_d_camXs = tracking_utils.geom.bilinear_sample2d(depth, query_xy_camXs[:,:,0], query_xy_camXs[:,:,1])
        query_xyd_camXs = torch.cat([query_xy_camXs, query_d_camXs.permute(0,2,1)], dim=2)
        query_xyz_camXs_cycle = tracking_utils.geom.xyd2pointcloud(query_xyd_camXs, pix_T_camXs[query_frame_id:query_frame_id+1])
        reproj_dist = torch.norm(query_xyz_camXs - query_xyz_camXs_cycle, dim=-1)
        visible = (reproj_dist < 0.1).reshape(-1).cpu().numpy() # N

        query_img = rgb_camXs[query_frame_id] # 3 x H x W
        nn_img = rgb_camXs[nn_frame_id] # 3 x H x W

        # TODO: reshape/crop the images. now just concat raw image together
        '''
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
        '''

            # query_img = tvtf.rotate(query_img, theta)
            # # adjust labels accordingly
            # query_keypoints_xy = tvtf.rotate(torch.tensor(query_keypoints_xy), theta).numpy()
        # if random.random() < self.rotation_chance:
        #     # theta = np.random.uniform(low=-1, high=1) * self.max_rotation
        #     theta=30
        #     nn_img = tvtf.rotate(nn_img, theta)
        #     nn_keypoints_xy = tvtf.rotate(torch.tensor(nn_keypoints_xy), theta).numpy()

        sbs_img = torch.cat([query_img, nn_img], axis=-1) # 3 x H x 2*W


        corrs = np.concatenate([query_keypoints_xy, nn_keypoints_xy], axis=1) # N x 4, x1y1x2y2
        mask_query = np.logical_and(np.logical_and(query_keypoints_xy[:,0]>0, query_keypoints_xy[:,0]<W), np.logical_and(query_keypoints_xy[:,1]>0, query_keypoints_xy[:,1]<H))
        mask_nn = np.logical_and(np.logical_and(nn_keypoints_xy[:,0]>0, nn_keypoints_xy[:,0]<W), np.logical_and(nn_keypoints_xy[:,1]>0, nn_keypoints_xy[:,1]<H))
        mask = np.logical_and(mask_nn, mask_query)
        corrs = corrs[mask]

        # if corrs.shape[0] < self.num_kp:
        if len(corrs) == 0:
            # print('bad example')
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        corrs = self._trim_corrs(corrs)

        # # for cv2 vis
        sbs_img_np = sbs_img.cpu().numpy()
        sbs_img_np = (np.transpose(sbs_img_np, (1, 2, 0))*255.0).astype(np.uint8).copy()
        for i in range(len(corrs)):
            corr = corrs[i] # 4
            if visible[i]:
                color = (0,255,0)
            else:
                color = (0,0,255)
            sbs_img_np = cv2.line(sbs_img_np, (int(corr[0]), int(corr[1])), (int(corr[2])+W, int(corr[3])), color, 1)

        cv2.imwrite('corr.png', sbs_img_np[..., [2,1,0]]) # rgb->bgr
        import ipdb; ipdb.set_trace()
        #time.sleep(1)
        #assert(False)

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

    def __getitem__(self, index):
        assert self.opt.k_size == 1

        sample = self.dataset[index]
        # if self.need_rotation:
        #     query_cap, nn_cap = self.augment_with_rotation(query_cap, nn_cap)

        # randomly select a query frame and a nn frame 
        pix_T_camXs = sample['pix_T_camXs'].cuda() # S x 4 x 4
        rgb_camXs = sample['rgb_camXs'].cuda() # S x 3 x H x W
        xyz_camXs = sample['xyz_camXs'].cuda() # S x N x 3
        origin_T_camXs = sample['world_T_camXs'].cuda() # S x 4 x 4
        scorelist = sample['scorelist_s'].cuda() # S x K
        lrtlist_camXs = sample['lrtlist_camXs'].cuda() # S x K x 19
        S, _, H, W = rgb_camXs.shape
        _, K = scorelist.shape

        rgb_camXs += .5 # range [0,1]

        # only take points belong to one object
        # obj_id = np.random.choice(int(scorelist[0].sum().item()))

        occluded = torch.zeros_like(sample['scorelist_s'])

        for obj_id in range(10):
            if scorelist[0, obj_id] == 0:
                continue

            nn_frame_id = 0
            inb = tracking_utils.geom.get_pts_inbound_lrt(xyz_camXs[nn_frame_id:nn_frame_id+1], lrtlist_camXs[nn_frame_id:nn_frame_id+1, obj_id], add_pad=0).reshape(-1) # N
            while inb.sum() < 50:
                occluded[nn_frame_id, obj_id] = 1
                nn_frame_id += 1
                if nn_frame_id == S:
                    break
                inb = tracking_utils.geom.get_pts_inbound_lrt(xyz_camXs[nn_frame_id:nn_frame_id+1], lrtlist_camXs[nn_frame_id:nn_frame_id+1, obj_id], add_pad=0).reshape(-1)
            
            if nn_frame_id == S:
                continue
            elif nn_frame_id == S-1:
                continue
            nn_xyz_camXs = xyz_camXs[nn_frame_id:nn_frame_id+1, inb] # 1, N, 3
            nn_xy_camXs = tracking_utils.geom.camera2pixels(nn_xyz_camXs, pix_T_camXs[nn_frame_id:nn_frame_id+1])

            nn_keypoints_x = nn_xy_camXs[..., 0] # 1 x N
            nn_keypoints_y = nn_xy_camXs[..., 1] # 1 x N, in image coord

            keypoints_xy = []
            visibles = []
            for query_frame_id in range(nn_frame_id+1, S):
                # transform the xyzs into query frame, to find correspondence
                origin_T_nn = origin_T_camXs[nn_frame_id:nn_frame_id+1]
                origin_T_query = origin_T_camXs[query_frame_id:query_frame_id+1]

                # if index == 8949 and obj_id == 4 and query_frame_id == 2:
                #     import ipdb; ipdb.set_trace()

                query_T_nn_cam = torch.matmul(tracking_utils.geom.safe_inverse(origin_T_query), origin_T_nn)

                mat1 = lrtlist_camXs[nn_frame_id, obj_id][3:].reshape(4,4)
                mat2 = lrtlist_camXs[query_frame_id, obj_id][3:].reshape(4,4)
                query_T_nn_obj = torch.matmul(mat2, mat1.inverse()).unsqueeze(0)

                query_T_nn = torch.matmul(query_T_nn_obj, query_T_nn_cam)

                query_xyz_camXs = tracking_utils.geom.apply_4x4(query_T_nn, nn_xyz_camXs)

                query_xy_camXs = tracking_utils.geom.camera2pixels(query_xyz_camXs, pix_T_camXs[query_frame_id:query_frame_id+1]) # 

                nn_keypoints_xy = nn_xy_camXs[0].cpu().numpy() # N x 2
                query_keypoints_xy = query_xy_camXs[0].cpu().numpy() # N x 2

                # obtain visible / non-visible
                depth, valid = tracking_utils.geom.create_depth_image(pix_T_camXs[query_frame_id:query_frame_id+1], xyz_camXs[query_frame_id:query_frame_id+1], H, W)
                query_d_camXs = tracking_utils.geom.bilinear_sample2d(depth, query_xy_camXs[:,:,0], query_xy_camXs[:,:,1])
                query_xyd_camXs = torch.cat([query_xy_camXs, query_d_camXs.permute(0,2,1)], dim=2)
                query_xyz_camXs_cycle = tracking_utils.geom.xyd2pointcloud(query_xyd_camXs, pix_T_camXs[query_frame_id:query_frame_id+1])
                reproj_dist = torch.norm(query_xyz_camXs - query_xyz_camXs_cycle, dim=-1)
                visible = (reproj_dist < 0.1).reshape(-1).cpu().numpy() # N

                query_img = rgb_camXs[query_frame_id] # 3 x H x W
                nn_img = rgb_camXs[nn_frame_id] # 3 x H x W

                if query_frame_id == 1:
                    keypoints_xy.append(nn_keypoints_xy)
                    keypoints_xy.append(np.copy(query_keypoints_xy))
                    visibles.append(np.copy(visible))
                else:
                    keypoints_xy.append(np.copy(query_keypoints_xy))
                    visibles.append(np.copy(visible))

                if visible.sum() == np.prod(visible.shape):
                    occluded[query_frame_id, obj_id] = 0
                else:
                    occluded[query_frame_id, obj_id] = 1
                    rgb_camXs[query_frame_id, :, np.clip(query_keypoints_xy[:,1], 0, W-1).astype(int), np.clip(query_keypoints_xy[:,0], 0, H-1).astype(int)] = 0


            keypoints_xy = np.stack(keypoints_xy, 0)
            visibles = np.stack(visibles, 0)
            mask = np.logical_and(((keypoints_xy[:,:,0] > 0) * (keypoints_xy[:,:,0] < W)).sum(0) == 6, ((keypoints_xy[:,:,1] > 0) * (keypoints_xy[:,:,1] < H)).sum(0) == 6)
            keypoints_xy = keypoints_xy[:,mask]
            visibles = visibles[:,mask]


            '''
            mask = np.random.choice(keypoints_xy.shape[1], 100)
            keypoints_xy = keypoints_xy[:,mask]
            visibles = visibles[:,mask]
            '''

        '''
        # # for cv2 vis
        sbs_img_np = (np.concatenate([rgb.cpu().numpy() for rgb in rgb_camXs], 2).transpose((1,2,0)) * 255).astype(np.uint8)

        kp_id = np.random.choice(keypoints_xy.shape[1])
        for i in range(1, S):
            x1 = keypoints_xy[i-1, kp_id, 0] + (i-1) * W
            x2 = keypoints_xy[i, kp_id, 0] + i * W
            y1 = keypoints_xy[i-1, kp_id, 1]
            y2 = keypoints_xy[i, kp_id, 1]
            if visibles[i-1, kp_id]:
                color = (0,255,0)
            else:
                color = (0,0,255)
            sbs_img_np = cv2.line(sbs_img_np, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

        cv2.imwrite('vis_cotr/corr{0}.png'.format(self.vis_count), sbs_img_np[..., [2,1,0]]) # rgb->bgr
        self.vis_count += 1
        if self.vis_count > 20:
            import sys; sys.exit()
        #time.sleep(1)
        #assert(False)
        '''

        self.occlusion[sample['filename']] = occluded.cpu().numpy()
        print(len(self.occlusion))

        return 1

    
# class COTRZoomDataset(COTRDataset):
#     def __init__(self, opt, dataset_type: str):
#         assert opt.crop_cam in ['no_crop', 'crop_center']
#         assert opt.use_ram == False
#         super().__init__(opt, dataset_type)
#         self.zoom_start = opt.zoom_start
#         self.zoom_end = opt.zoom_end
#         self.zoom_levels = opt.zoom_levels
#         self.zoom_jitter = opt.zoom_jitter
#         self.zooms = np.logspace(np.log10(opt.zoom_start),
#                                  np.log10(opt.zoom_end),
#                                  num=opt.zoom_levels)

#     def get_corrs(self, from_cap, to_cap, reduced_size=None):
#         from_y, from_x = np.where(from_cap.depth_map > 0)
#         from_y, from_x = from_y[..., None], from_x[..., None]
#         if reduced_size is not None:
#             filter_idx = np.random.choice(from_y.shape[0], reduced_size, replace=False)
#             from_y, from_x = from_y[filter_idx], from_x[filter_idx]
#         from_z = from_cap.depth_map[np.floor(from_y).astype('int'), np.floor(from_x).astype('int')]
#         from_xy = np.concatenate([from_x, from_y], axis=1)
#         from_3d_world, valid_index_1 = pcd_projector.PointCloudProjector.pcd_2d_to_pcd_3d_np(from_xy, from_z, from_cap.pinhole_cam.intrinsic_mat, motion=from_cap.cam_pose.camera_to_world, return_index=True)

#         to_xyz, valid_index_2 = pcd_projector.PointCloudProjector.pcd_3d_to_pcd_2d_np(
#             from_3d_world,
#             to_cap.pinhole_cam.intrinsic_mat,
#             to_cap.cam_pose.world_to_camera[0:3, :],
#             to_cap.image.shape[:2],
#             keep_z=True,
#             crop=True,
#             filter_neg=True,
#             norm_coord=False,
#             return_index=True,
#         )

#         to_xy = to_xyz[:, 0:2]
#         to_z_proj = to_xyz[:, 2:3]
#         to_z = to_cap.depth_map[np.floor(to_xy[:, 1:2]).astype('int'), np.floor(to_xy[:, 0:1]).astype('int')]
#         mask = (abs(to_z - to_z_proj) < 0.5)[:, 0]
#         if mask.sum() > 0:
#             return np.concatenate([from_xy[valid_index_1][valid_index_2][mask], to_xy[mask]], axis=1)
#         else:
#             return None

#     def get_seed_corr(self, from_cap, to_cap, max_try=100):
#         seed_corr = self.get_corrs(from_cap, to_cap, reduced_size=max_try)
#         if seed_corr is None:
#             return None
#         shuffle = np.random.permutation(seed_corr.shape[0])
#         seed_corr = np.take(seed_corr, shuffle, axis=0)
#         return seed_corr[0]

#     def get_zoomed_cap(self, cap, pos, scale, jitter):
#         patch = inference_helper.get_patch_centered_at(cap.image, pos, scale=scale, return_content=False)
#         patch = inference_helper.get_patch_centered_at(cap.image,
#                                                   pos + np.array([patch.w, patch.h]) * np.random.uniform(-jitter, jitter, 2),
#                                                   scale=scale,
#                                                   return_content=False)
#         zoom_config = CropCamConfig(x=patch.x,
#                                     y=patch.y,
#                                     w=patch.w,
#                                     h=patch.h,
#                                     out_w=constants.MAX_SIZE,
#                                     out_h=constants.MAX_SIZE,
#                                     orig_w=cap.shape[1],
#                                     orig_h=cap.shape[0])
#         zoom_cap = capture.crop_capture(cap, zoom_config)
#         return zoom_cap

#     def __getitem__(self, index):
#         assert self.opt.k_size == 1
#         query_cap, nn_caps = self.sfm_dataset.get_query_with_knn(index)
#         nn_cap = nn_caps[0]
#         if self.need_rotation:
#             query_cap, nn_cap = self.augment_with_rotation(query_cap, nn_cap)

        

#         # find seed
#         seed_corr = self.get_seed_corr(nn_cap, query_cap)
#         if seed_corr is None:
#             return self.__getitem__(random.randint(0, self.__len__() - 1))

#         # crop cap
#         s = np.random.choice(self.zooms)
#         nn_zoom_cap = self.get_zoomed_cap(nn_cap, seed_corr[:2], s, 0)
#         query_zoom_cap = self.get_zoomed_cap(query_cap, seed_corr[2:], s, self.zoom_jitter)
#         assert nn_zoom_cap.shape == query_zoom_cap.shape == (constants.MAX_SIZE, constants.MAX_SIZE)
#         corrs = self.get_corrs(query_zoom_cap, nn_zoom_cap)
#         if corrs is None or corrs.shape[0] < self.num_kp:
#             return self.__getitem__(random.randint(0, self.__len__() - 1))
#         shuffle = np.random.permutation(corrs.shape[0])
#         corrs = np.take(corrs, shuffle, axis=0)
#         corrs = self._trim_corrs(corrs)

#         # flip augmentation
#         if np.random.uniform() < 0.5:
#             corrs[:, 0] = constants.MAX_SIZE - 1 - corrs[:, 0]
#             corrs[:, 2] = constants.MAX_SIZE - 1 - corrs[:, 2]
#             sbs_img = two_images_side_by_side(np.fliplr(query_zoom_cap.image), np.fliplr(nn_zoom_cap.image))
#         else:
#             sbs_img = two_images_side_by_side(query_zoom_cap.image, nn_zoom_cap.image)

#         corrs[:, 2] += constants.MAX_SIZE
#         corrs /= np.array([constants.MAX_SIZE * 2, constants.MAX_SIZE, constants.MAX_SIZE * 2, constants.MAX_SIZE])
#         assert (0.0 <= corrs[:, 0]).all() and (corrs[:, 0] <= 0.5).all()
#         assert (0.0 <= corrs[:, 1]).all() and (corrs[:, 1] <= 1.0).all()
#         assert (0.5 <= corrs[:, 2]).all() and (corrs[:, 2] <= 1.0).all()
#         assert (0.0 <= corrs[:, 3]).all() and (corrs[:, 3] <= 1.0).all()
#         out = {
#             'image': tvtf.normalize(tvtf.to_tensor(sbs_img), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#             'corrs': torch.from_numpy(corrs).float(),
#         }
#         if self.bidirectional:
#             out['queries'] = torch.from_numpy(np.concatenate([corrs[:, :2], corrs[:, 2:]], axis=0)).float()
#             out['targets'] = torch.from_numpy(np.concatenate([corrs[:, 2:], corrs[:, :2]], axis=0)).float()
#         else:
#             out['queries'] = torch.from_numpy(corrs[:, :2]).float()
#             out['targets'] = torch.from_numpy(corrs[:, 2:]).float()

#         return out
