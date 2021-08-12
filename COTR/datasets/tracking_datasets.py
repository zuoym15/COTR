from numpy import random
from numpy.core.numeric import full
import torch
import numpy as np
import os
import scipy.ndimage
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random

# from torch._C import dtype, set_flush_denormal
# from detectron2.structures.masks import polygons_to_bitmask
import sys
sys.path.append('.')

from . import utils
from .utils import py
from .utils import geom
# from .utils import improc

# import utils.py
# import utils.geom
# import utils.improc

import glob
import json

def get_dataset(name, seqlen, shuffle=True, env='plate'):
    if name == 'manip':
        return ManipDataset(shuffle=shuffle, seqlen=seqlen, env=env)
    elif name == 'mcs':
        return McsDataset(shuffle=shuffle, seqlen=seqlen)
    elif name == 'cater':
        return CaterDataset(shuffle=shuffle, seqlen=seqlen, env=env)
    elif name == 'arrow':
        return ArrowDataset(shuffle=shuffle, seqlen=seqlen)
    else:
        assert "invalid dataset: %s" % name


class CaterBackgroundDataset(torch.utils.data.Dataset):
    def __init__(self, image_location, dataset_location, crop_size=64, mod='ab', num_videos=10):
        records = [filename for filename in sorted(os.listdir(dataset_location)) if mod in filename]
        records = records[:num_videos] # debug

        nRecords = len(records)
        print('found %d records in %s' % (nRecords, dataset_location))
        
        nCheck = np.min([nRecords, 1000])
        for record in records[:nCheck]:
            assert os.path.isfile(dataset_location + '/' + record)
        print('checked the first %d, and they seem to be real files' % (nCheck))

        self.records = records
        self.dataset_location = dataset_location
        self.image_location = image_location
        self.crop_size = crop_size
        self.mod = mod

        all_rgbs = dict()
        print('loading anns and rgbs into memory...')
        for i, record in enumerate(self.records):
            raw_data_filename = record.replace('_'+mod, '')

            rgb = dict(np.load(os.path.join(self.image_location, raw_data_filename), allow_pickle=True))['rgb_camXs'] # S x H x W x 3
            rgb = np.transpose(rgb, (0, 3, 1, 2)) # S x 3 x H x W
            rgb = utils.improc.preprocess_color(rgb) 
            rgb = torch.from_numpy(rgb)

            all_rgbs[raw_data_filename] = rgb # S x 3 x H x W

            if (i+1) % (len(self.records) // 10) == 0: # this is slow, so we add a progress bar
                print('finished %d/%d' % (i+1, len(self.records)))

        self.all_rgbs = all_rgbs

    def __getitem__(self, index):
        # read detections
        det_filename = self.records[index]
        det_fullpath = os.path.join(self.dataset_location, det_filename)
        det_meta = np.load(det_fullpath, allow_pickle=True)
        det_scores = det_meta['scorelist']
        det_boxes = det_meta['boxlist_2d']
        seglist = det_meta['seglist']
        det_scores_binary = det_scores > 0

        # read images
        traj_filename = det_filename.replace('_'+self.mod, '')
        rgb = self.all_rgbs[traj_filename]
        _, _, H, W = rgb.shape
        S, N = det_scores_binary.shape

        full_mask = np.zeros((N, 1, H, W))

        timestep = np.random.randint(0, S)
        for obj_id in range(N):
            if det_scores_binary[timestep, obj_id]:
                (tl_x, tl_y, region_w, region_h), rle = seglist[timestep][obj_id]
                cropped_mask = self.rle_to_mask(rle, region_w, region_h)
                full_mask[obj_id, 0, tl_y:tl_y+region_h, tl_x:tl_x+region_w] = cropped_mask

        full_mask = np.sum(full_mask, axis=0, keepdims=True)
        full_mask = np.clip(full_mask, 0.0, 1.0)
        full_mask = 1.0 - full_mask # take inverse. 1.0 for background areas

        full_mask = torch.from_numpy(full_mask).float()
        rgb_frame = rgb[timestep].unsqueeze(0)

        full_mask = utils.improc.erode2d(full_mask, times=2, device='cpu')

        full_mask = F.interpolate(full_mask, size=(self.crop_size, self.crop_size)).squeeze(0)
        rgb_frame = F.interpolate(rgb_frame, size=(self.crop_size, self.crop_size)).squeeze(0)


        return {'rgb': rgb_frame, 'mask': full_mask}

    def rle_to_mask(self, rle, width, height):
        """
        rle: input rle mask encoding
        each evenly-indexed element represents number of consecutive 0s
        each oddly indexed element represents number of consecutive 1s
        width and height are dimensions of the mask
        output: 2-D binary mask
        """
        # allocate list of zeros
        v = [0] * (width * height)

        # set id of the last different element to the beginning of the vector
        idx_ = 0
        for i in range(len(rle)):
            if i % 2 != 0:
                # write as many 1s as RLE says (zeros are already in the vector)
                for j in range(rle[i]):
                    v[idx_+j] = 1
            idx_ += rle[i]

        # reshape vector into 2-D mask
        # return np.reshape(np.array(v, dtype=np.uint8), (height, width)) # numba bug / not supporting np.reshape
        return np.array(v, dtype=np.float32).reshape((height, width))

    def __len__(self):
        return len(self.records)

class CaterObjCropDataset(torch.utils.data.Dataset):
    def __init__(self, image_location, dataset_location, crop_size=32, mod='ab', num_videos=10, do_masking=True, padding=0):
        records = [filename for filename in sorted(os.listdir(dataset_location)) if mod in filename]
        records = records[:num_videos] # debug

        nRecords = len(records)
        print('found %d records in %s' % (nRecords, dataset_location))
        
        nCheck = np.min([nRecords, 1000])
        for record in records[:nCheck]:
            assert os.path.isfile(dataset_location + '/' + record)
        print('checked the first %d, and they seem to be real files' % (nCheck))

        self.records = records
        self.dataset_location = dataset_location
        self.image_location = image_location
        self.crop_size = crop_size
        self.mod = mod

        # all_rgbs = dict()
        all_rgbs = []
        all_masks = []

        # flatten everything
        print('loading anns and rgbs into memory...')
        for i, record in enumerate(self.records):
            raw_data_filename = record[:-7]+'.npz'

            rgb = dict(np.load(os.path.join(self.image_location, raw_data_filename), allow_pickle=True))['rgb_camXs'] # S x H x W x 3
            
            # all_rgbs[raw_data_filename] = rgb # S x H x W x 3

            data = np.load(os.path.join(dataset_location, record), allow_pickle=True)
            data = dict(data)
            boxlist_2d = data['boxlist_2d']
            scorelist = data['scorelist']
            seglist = data['seglist']


            # for some reason, frame0 doesn't have anything; just 
            boxlist_2d = boxlist_2d[2:3]
            scorelist = scorelist[2:3]
            seglist = seglist[2:3]
            rgb = rgb[2:3]
            
            rgb = np.transpose(rgb, (0, 3, 1, 2)) # S x 3 x H x W
            rgb = utils.improc.preprocess_color(rgb) 
            rgb = torch.from_numpy(rgb)

            _, _, H, W = rgb.shape
            

            # print('scorelist', scorelist)
            
            S, N, _ = boxlist_2d.shape
            for n in range(N):
                valid = scorelist[:,n] > 0 # S
                if np.sum(valid) > 0:
                    frame_ids = (np.where(valid)[0])

                    full_mask = np.zeros((np.sum(valid), 1, H, W))
            
                    for j, frame_id in enumerate(frame_ids):
                        (tl_x, tl_y, region_w, region_h), rle = seglist[frame_id][n]
                        cropped_mask = self.rle_to_mask(rle, region_w, region_h)
                        full_mask[j, 0, tl_y:tl_y+region_h, tl_x:tl_x+region_w] = cropped_mask

                    rgb_obj = rgb[valid] # s_valid x 3 x H x W
                    box = torch.from_numpy(boxlist_2d[valid, n, ]) # s_valid x 4
                    x1, y1, x2, y2 = torch.unbind(box, dim=1)
                    box = torch.stack([y1-padding, x1-padding, y2+padding, x2+padding], dim=1)

                    full_mask = torch.from_numpy(full_mask).float()

                    # do cropping
                    crops = utils.geom.crop_and_resize(rgb_obj, box, self.crop_size, self.crop_size, box2d_is_normalized=False)
                    masks = utils.geom.crop_and_resize(full_mask, box, self.crop_size, self.crop_size, box2d_is_normalized=False)

                    all_rgbs.append(crops)
                    all_masks.append(masks)

            if (i+1) % (len(self.records) // 10) == 0: # this is slow, so we add a progress bar
                print('finished %d/%d' % (i+1, len(self.records)))

        self.all_rgbs = torch.cat(all_rgbs)
        self.all_masks = torch.cat(all_masks)

        self.num_crops = len(self.all_rgbs)

        print('found %d crops' % self.num_crops)

    def __getitem__(self, index):
        # read detections
        cropA = self.all_rgbs[index]
        maskA = self.all_masks[index]

        return {'rgb': cropA, 'mask': maskA, 'idx': index}

    def rle_to_mask(self, rle, width, height):
        """
        rle: input rle mask encoding
        each evenly-indexed element represents number of consecutive 0s
        each oddly indexed element represents number of consecutive 1s
        width and height are dimensions of the mask
        output: 2-D binary mask
        """
        # allocate list of zeros
        v = [0] * (width * height)

        # set id of the last different element to the beginning of the vector
        idx_ = 0
        for i in range(len(rle)):
            if i % 2 != 0:
                # write as many 1s as RLE says (zeros are already in the vector)
                for j in range(rle[i]):
                    v[idx_+j] = 1
            idx_ += rle[i]

        # reshape vector into 2-D mask
        # return np.reshape(np.array(v, dtype=np.uint8), (height, width)) # numba bug / not supporting np.reshape
        return np.array(v, dtype=np.float32).reshape((height, width))

    def __len__(self):
        return self.num_crops


class CaterBoxDataset(torch.utils.data.Dataset):
    def __init__(self, image_location, dataset_location, crop_size=32, mod='aa', num_videos=10):
        records = [filename for filename in sorted(os.listdir(dataset_location)) if mod in filename]
        # records = records[:num_videos]  # debug

        nRecords = len(records)
        print('found %d records in %s' % (nRecords, dataset_location))

        nCheck = np.min([nRecords, 1000])
        for record in records[:nCheck]:
            assert os.path.isfile(dataset_location + '/' + record)
        print('checked the first %d, and they seem to be real files' % (nCheck))

        self.records = records
        self.dataset_location = dataset_location
        self.image_location = image_location
        self.crop_size = crop_size
        self.mod = mod

        for i, record in enumerate(self.records):
            raw_data_filename = record.replace('_'+mod, '')

            rgb = dict(np.load(os.path.join(self.image_location, raw_data_filename), allow_pickle=True))['rgb_camXs'] # S x H x W x 3
            rgb = np.transpose(rgb, (0, 3, 1, 2)) # S x 3 x H x W
            _, _, self.H, self.W = rgb.shape
            break

    def __getitem__(self, index):
        # read detections
        det_filename = self.records[index]
        det_fullpath = os.path.join(self.dataset_location, det_filename)
        det_meta = np.load(det_fullpath, allow_pickle=True)
        det_scores = det_meta['scorelist']
        det_boxes = det_meta['boxlist_2d']
        det_scores_binary = det_scores > 0

        obj_id = np.random.randint(det_scores_binary.shape[1])
        timestep = np.random.choice(np.nonzero(det_scores_binary[:, obj_id])[0])

        boxA = torch.from_numpy(det_boxes[timestep, obj_id]).unsqueeze(0)  # (1, 4)
        c = (boxA[:, :2] + boxA[:, 2:]) / 2
        l = torch.abs(boxA[:, 2:] - boxA[:, :2])
        c[:, 0] /= self.W
        c[:, 1] /= self.H
        l[:, 0] /= self.W
        l[:, 1] /= self.H
        return torch.cat([c, l], dim=1)

    def __len__(self):
        return len(self.records)

class CaterCopyPasteDataset(torch.utils.data.Dataset):
    def __init__(self, ann_path, raw_data_path, H=240, W=320, max_num_objs=10, min_num_objs=5, mod='ab', width_range=[15., 60.]):
        self.ann_path = ann_path
        self.raw_data_path = raw_data_path

        self.min_num_objs = min_num_objs
        self.max_num_objs = max_num_objs
        self.width_range = width_range

        self.H = H
        self.W = W

        all_boxes = []
        all_data_paths = []
        all_frame_ids = []
        all_rles = []
        all_rgbs = dict() # load in __init__ to accelerate

        self.all_records = os.listdir(ann_path)
        self.all_records = [record for record in self.all_records if mod in record] # filter

        self.all_cameras = ['Camera_1', 'Camera_2', 'Camera_3', 'Camera_4', 'Camera_5', 'Camera_6']
        records_filtered = []

        for cam_name in self.all_cameras:
            cam_record = [record for record in self.all_records if cam_name in record] # filter
            random.shuffle(cam_record)

            records_filtered.extend(cam_record[:5]) # 5 videos for each view

        self.all_records = records_filtered

        # self.all_records = self.all_records[:30] # too many is not necessary

        print('found %d records' % len(self.all_records))

        # flatten everything
        print('loading anns and rgbs into memory...')
        for i, record in enumerate(self.all_records):
            raw_data_filename = record[:-7]+'.npz'

            rgb = dict(np.load(os.path.join(self.raw_data_path, raw_data_filename), allow_pickle=True))['rgb_camXs'] # S x H x W x 3
            
            all_rgbs[raw_data_filename] = rgb # S x H x W x 3

            data = np.load(os.path.join(ann_path, record), allow_pickle=True)
            data = dict(data)
            boxlist_2d = data['boxlist_2d']
            scorelist = data['scorelist']
            seglist = data['seglist']
            S, N, _ = boxlist_2d.shape
            for n in range(N):
                valid = scorelist[:,n] > 0 # S
                if np.sum(valid) > 0:
                    frame_ids = (np.where(valid)[0])
                    all_frame_ids.append(frame_ids)

                    box = boxlist_2d[valid, n, ] # s_valid x 4
                    all_boxes.append(box)

                    all_data_paths.append(np.array([raw_data_filename]*np.sum(valid)))

                    # print(len(box))
                    # print(len(np.where(valid)[0]))
                    # print(len(all_data_paths[-1]))
                    for frame_id in frame_ids:
                        rle = seglist[frame_id][n]
                        all_rles.append(rle)

            if (i+1) % (len(self.all_records) // 10) == 0: # this is slow, so we add a progress bar
                print('finished %d/%d' % (i+1, len(self.all_records)))

                    


        self.all_boxes = np.concatenate(all_boxes)
        self.all_data_paths = np.concatenate(all_data_paths)
        self.all_frame_ids = np.concatenate(all_frame_ids)
        self.all_rgbs = all_rgbs
        self.all_rles = all_rles

        
        self.num_crops = len(self.all_boxes)
        assert(len(self.all_frame_ids)==self.num_crops)
        assert(len(self.all_data_paths)==self.num_crops)
        assert(len(self.all_rles)==self.num_crops)

        print('found %d crops' % self.num_crops)

        self.canvases = []

       

        for cam_name in self.all_cameras:
            cam_record = [record for record in self.all_records if cam_name in record] # filter
            print('num of %s videos' % cam_name, len(cam_record))

            if len(cam_record) > 0:
                rgb_to_process = []
                for record in cam_record:
                    raw_data_filename = record[:-7]+'.npz'
                    rgb_to_process.append(all_rgbs[raw_data_filename]) # S x H x W x 3

                rgb_to_process = np.concatenate(rgb_to_process, axis=0) # NS x H x W x 3
                canvas = np.median(rgb_to_process, axis=0).transpose(2, 0, 1)
                canvas = utils.improc.preprocess_color(canvas)

                self.canvases.append(canvas)
    
    def __getitem__(self, index):
        # generate a random scene
        # let's create a background
        # ann_filename = self.all_records[index]
        # raw_data_filename = ann_filename[:-7]+'.npz'

        # bkg_data = self.all_rgbs[raw_data_filename]
        # canvas = np.median(bkg_data, axis=0) # 3 x H x W
        # canvas = np.zeros((3, self.H, self.W))
        canvas_id = np.random.randint(0, len(self.canvases))
        canvas = self.canvases[canvas_id]

        # now let's paste things onto the canvas
        # TODO: add all kinds of aug (size, position, colors)

        num_objs = np.random.randint(self.min_num_objs, self.max_num_objs+1)

        # choose n objs from the list
        obj_ids = np.random.choice(np.arange(self.num_crops), size=num_objs, replace=False)

        boxlist_2d = np.zeros((self.max_num_objs, 4))
        scorelist = np.zeros(self.max_num_objs)
        full_masks = np.zeros((self.max_num_objs, 1, self.H, self.W))

        for i, obj_id in enumerate(obj_ids):
            obj_filename = self.all_data_paths[obj_id]
            frame_id = self.all_frame_ids[obj_id]
            box = np.round(self.all_boxes[obj_id]) # 4
            rgb = (self.all_rgbs[obj_filename])[frame_id] # H x W x 3

            # some layers
            full_mask = np.zeros((1, self.H, self.W))
            new_rgb = np.zeros((3, self.H, self.W))

            # generate mask
            (tl_x, tl_y, region_w, region_h), rle = self.all_rles[obj_id]

            cropped_mask = self.rle_to_mask(rle, region_w, region_h)
            full_mask[0, tl_y:tl_y+region_h, tl_x:tl_x+region_w] = cropped_mask

                    
            x1, y1, x2, y2 = box
            x, y, w, h = (x1+x2)/2., (y1+y2)/2., (x2-x1), (y2-y1)

            # do zooming
            zoom_factor = np.random.uniform(max(0.5, self.width_range[0]/w), min(1.5, self.width_range[1]/w))

            w, h = zoom_factor*w, zoom_factor*h # keep centroid unmoved

            # let's pick a new location for this box. use reject samplint to avoid large overlap
            ok = False
            
            for _ in range(10):
                if ok:
                    break
                # randomly pick a location
                x_new = np.random.uniform(10. + w/2., self.W-w/2. - 10.)
                y_new = np.random.uniform(10. + h/2., self.H-h/2. - 10.)

                # count the area already occupied in this box (like iou)
                x1, y1, x2, y2 = x_new - w/2., y_new - h/2., x_new + w/2., y_new + h/2.

                if i == 0: # the first one is always ok
                    ok=True
                    break

                area_occupied = np.sum(full_masks[:i, 0, round(y1):round(y2), round(x1):round(x2)], axis=(1,2)) # i-1
                area_full = np.sum(full_masks[:i, 0], axis=(1,2)) # i

                if np.all(area_occupied < 0.3 * area_full): # a good location
                    ok = True

            if not ok:
                continue # give up putting this obj
                
            x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

            # color jittering
            color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)
            rgb_pil = Image.fromarray(rgb)
            rgb_pil = color_jitter(rgb_pil)
            rgb = np.array(rgb_pil)

            rgb = np.transpose(rgb, (2, 0, 1)) # 3 x H x W
            rgb = utils.improc.preprocess_color(rgb)

            # zoom img and mask
            rgb = scipy.ndimage.zoom(rgb, (1, zoom_factor, zoom_factor), order=0)
            full_mask = scipy.ndimage.zoom(full_mask, (1, zoom_factor, zoom_factor), order=0) 

            # paste into a new layer
            try:
                full_masks[i, :, y1:y2, x1:x2] = full_mask[:, round(box[1]*zoom_factor):round(box[1]*zoom_factor)+y2-y1, round(box[0]*zoom_factor):round(box[0]*zoom_factor)+x2-x1]
                new_rgb[:, y1:y2, x1:x2] = rgb[:, round(box[1]*zoom_factor):round(box[1]*zoom_factor)+y2-y1, round(box[0]*zoom_factor):round(box[0]*zoom_factor)+x2-x1]
            except:
                continue # give up putting this obj
        
            # handle mask occlusions
            if i > 0:
                full_masks[:i] *= (1. - full_masks[i:i+1])
            
            # paste
            # canvas[:, y1:y2, x1:x2] = rgb[frame_id, :, y1:y2, x1:x2]
            # canvas = (1. - full_masks[i]) * canvas + full_masks[i] * rgb
            canvas = (1. - full_masks[i]) * canvas + full_masks[i] * new_rgb

            scorelist[i] = 1.0
            boxlist_2d[i, :] = np.array([y1/self.H, x1/self.W, y2/self.H, x2/self.W])

        return canvas, full_masks, boxlist_2d, scorelist

    def rle_to_mask(self, rle, width, height):
        """
        rle: input rle mask encoding
        each evenly-indexed element represents number of consecutive 0s
        each oddly indexed element represents number of consecutive 1s
        width and height are dimensions of the mask
        output: 2-D binary mask
        """
        # allocate list of zeros
        v = [0] * (width * height)

        # set id of the last different element to the beginning of the vector
        idx_ = 0
        for i in range(len(rle)):
            if i % 2 != 0:
                # write as many 1s as RLE says (zeros are already in the vector)
                for j in range(rle[i]):
                    v[idx_+j] = 1
            idx_ += rle[i]

        # reshape vector into 2-D mask
        # return np.reshape(np.array(v, dtype=np.uint8), (height, width)) # numba bug / not supporting np.reshape
        return np.array(v, dtype=np.float32).reshape((height, width))

    def __len__(self):
        return len(self.all_records)

class CaterMotionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_loc, min_length=5, max_length=30, W=320, H=240, noise_sigma=0.005, pad_input=True):
        self.dataset_loc = dataset_loc
        self.min_length = min_length
        self.max_length = max_length
        self.noise_sigma = noise_sigma
        self.pad_input = pad_input

        all_trajs = []

        all_records = os.listdir(dataset_loc)
        print('found %d records' % len(all_records))

        for record in all_records:
            data = np.load(os.path.join(dataset_loc, record), allow_pickle=True)
            data = dict(data)
            boxlist_2d = data['boxlist_2d']
            scorelist = data['scorelist']
            S, N, _ = boxlist_2d.shape
            for n in range(N):
                valid = scorelist[:,n] > 0 # S
                if np.sum(valid) > min_length:
                    box = boxlist_2d[valid, n, ] # s_valid x 4
                    x = ((box[:,0] + box[:,2]) / 2) / float(W) # normalize to [0,1]
                    y = ((box[:,1] + box[:,3]) / 2) / float(H) # normalize to [0,1]

                    all_trajs.append(np.stack([x, y], axis=1)) # s_valid x 2

        self.all_trajs = all_trajs
        print('found %d trajs' % len(all_trajs))

    def __getitem__(self, index):
        traj = self.all_trajs[index] # s_valid x 2
        return_len = np.random.randint(low=self.min_length, high=(min(self.max_length, len(traj))))

        history_pos = np.zeros((self.max_length, 2))

        start_pos = np.random.randint(low=0, high=len(traj) - return_len)
        # history_pos = traj[start_pos:start_pos+return_len-1, :]
        history_pos[:return_len-1] = traj[start_pos:start_pos+return_len-1, :]
        current_pos = traj[start_pos+return_len-1:start_pos+return_len, :]

        if self.noise_sigma > 0.0:
            history_pos[:return_len-1] += np.random.normal(scale=self.noise_sigma, size=(history_pos[:return_len-1]).shape)
            current_pos += np.random.normal(scale=self.noise_sigma, size=current_pos.shape)

        history_pos = torch.from_numpy(history_pos).float()
        current_pos = torch.from_numpy(current_pos).float()

        if self.pad_input:
            return history_pos, current_pos, torch.Tensor([return_len-1])
        else:
            return history_pos[:return_len-1], current_pos, torch.Tensor([return_len-1])

        

    def __len__(self):
        return len(self.all_trajs)


class SimpleManipDataset(torch.utils.data.Dataset):

    # what i want here is to return ALL FRAMES and ALL VIEWS
    
    def __init__(self, shuffle=True, env=''):

        # manipulation_dataset_location = "/projects/katefgroup/datasets/manipulation/raw/ae"
        manipulation_dataset_location = "/projects/katefgroup/datasets/manipulation/raw/ag"

        dataset_location = "%s" % manipulation_dataset_location

        glob_path = os.path.join(manipulation_dataset_location, "episode_[0-9][0-9][0-9][0-9]_%s*.npz" % env)
        print('glob_path', glob_path)
        records = glob.glob(glob_path)
        print('records', records)
        # records = [dataset_location + '/' + filename for filename in os.listdir(manipulation_dataset_location) if env==filename[:len(env)]]
        nRecords = len(records)
        print('found %d records in %s/%s' % (nRecords, dataset_location, env))
        nCheck = np.min([nRecords, 1000])
        for record in records[:nCheck]:
            assert os.path.isfile(record), 'Record at %s was not found' % record
        print('checked the first %d, and they seem to be real files' % (nCheck))

        self.records = records
        self.shuffle = shuffle

    def __getitem__(self, index):
        filename = self.records[index]
        d = dict(np.load(filename, allow_pickle=True))
        d['filename'] = filename
        return d
    
    def __len__(self):
        return len(self.records)
        # return 10

    def get_item_names(self):
        item_names = [
            'rgbd',
            'intrinsics',
            'bbox_center',
            'extrinsics',
            'bbox_extent',
            'bbox_rot',
            'robot_mask',
        ]
        return item_names


class ManipDataset(torch.utils.data.Dataset):
    def __init__(self, shuffle=True, seqlen=50, env='plate'):

        manipulation_dataset_location = "/projects/katefgroup/datasets/manipulation/raw/ag"

        dataset_location = "%s" % manipulation_dataset_location

        records = [dataset_location + '/' + filename for filename in sorted(os.listdir(manipulation_dataset_location))
                   if env in filename]
        nRecords = len(records)
        print('found %d records in %s' % (nRecords, dataset_location))
        nCheck = np.min([nRecords, 1000])
        for record in records[:nCheck]:
            assert os.path.isfile(record), 'Record at %s was not found' % record
        print('checked the first %d, and they seem to be real files' % (nCheck))

        self.records = records
        self.shuffle = shuffle
        self.seqlen = seqlen


    def __getitem__(self, index):
        filename = self.records[index]
        d = np.load(filename, allow_pickle=True)

        if not self.shuffle:
            n = 0  # choose a view
        else:
            n = np.random.randint(6)  # choose a view

        if self.shuffle:
            d = random_select_single(self.get_item_names(), d, num_samples=self.seqlen)
        else:
            d = non_random_select_single(self.get_item_names(), d, num_samples=self.seqlen)

        rgbd_cam = d['rgbd'][:, n]
        intrinsics = d['intrinsics'][:, n]
        bbox_center = d['bbox_center']  # S x 3
        world_T_cam = d['extrinsics'][:, n]  # S x N x 4 x 4
        lens = d['bbox_extent']  # S x 3
        bbox_rot = d['bbox_rot']  # S x 3 x 3
        robot_mask = d['robot_mask'][:, n]  # S x N x H x W x 1

        # print('robot_mask', robot_mask.shape)
        return_dict = {}

        pix_T_cam = np.eye(4)[np.newaxis, :, :].repeat(rgbd_cam.shape[0], axis=0)
        pix_T_cam[:, :3, :3] = intrinsics
        pix_T_cam = torch.from_numpy(pix_T_cam).float()

        rgb = rgbd_cam[:, :, :, 0:3]
        rgb = rgb.transpose(0, 3, 1, 2)
        rgb = torch.from_numpy(rgb)
        rgb = utils.improc.preprocess_color(rgb)

        depth = rgbd_cam[:, :, :, 3:]  # S x H x W x 1
        depth = depth.transpose(0, 3, 1, 2)
        depth = torch.from_numpy(depth)
        xyz_cam = utils.geom.depth2pointcloud(depth, pix_T_cam).float()  # S x N x 3

        bbox_center = torch.from_numpy(bbox_center)  # S x 3
        lens = torch.from_numpy(lens)  # S x 3
        bbox_rot = torch.from_numpy(bbox_rot)  # S x 3 x 3
        rt = utils.geom.merge_rt(bbox_rot, bbox_center)  # S x 4 x 4
        lrt = torch.cat([lens, rt.reshape(rgbd_cam.shape[0], -1)], dim=1).float()  # S x 19
        lrtlist = lrt.unsqueeze(1)
        world_T_cam = torch.from_numpy(world_T_cam).float()
        cam_T_world = utils.geom.safe_inverse(world_T_cam)
        lrtlist_camXs = utils.geom.apply_4x4_to_lrtlist(cam_T_world, lrtlist)

        robot_mask = torch.from_numpy(robot_mask)

        return_dict['rgb_camXs'] = rgb
        return_dict['pix_T_camXs'] = pix_T_cam
        return_dict['filename'] = filename
        return_dict['xyz_camXs'] = xyz_cam
        return_dict['lrtlist_camXs'] = lrtlist_camXs
        return_dict['scorelist_s'] = torch.ones_like(lrtlist_camXs[:, :, 0])
        return_dict['robot_mask'] = robot_mask
        return_dict['filename'] = filename
        return return_dict

    def __len__(self):
        return len(self.records)
        # return 10

    def get_item_names(self):
        item_names = [
            'rgbd',
            'intrinsics',
            'bbox_center',
            'extrinsics',
            'bbox_extent',
            'bbox_rot',
            'robot_mask',
        ]
        return item_names

class McsSingleDataset(torch.utils.data.Dataset):
    def __init__(self, shuffle=True, seqlen=50):

        mcs_data_mod = 'aj'
        mcs_dataset_location = "/projects/katefgroup/datasets/mcs/processed"

        trainset = "s%st" % (mcs_data_mod)
        dataset_location = "%s" % mcs_dataset_location
        dataset_path = '%s/%s.txt' % (dataset_location, trainset)

        print('dataset_path = %s' % dataset_path)
        with open(dataset_path) as f:
            content = f.readlines()
        dataset_location = dataset_path.split('/')[:-1]
        dataset_location = '/'.join(dataset_location)
        print('dataset_loc = %s' % dataset_location)
        
        records = [dataset_location + '/' + line.strip() for line in content]
        nRecords = len(records)
        print('found %d records in %s' % (nRecords, dataset_path))
        nCheck = np.min([nRecords, 1000])
        for record in records[:nCheck]:
            assert os.path.isfile(record), 'Record at %s was not found' % record
        print('checked the first %d, and they seem to be real files' % (nCheck))
 
        self.records = records
        self.shuffle = shuffle
        self.seqlen = seqlen

    def __getitem__(self, index):
        filename = self.records[index]
        d = np.load(filename, allow_pickle=True)
        d = dict(d)

        if self.shuffle:
            d = random_select_single(self.get_item_names(), d, num_samples=self.seqlen)
        else:
            d = non_random_select_single(self.get_item_names(), d, num_samples=self.seqlen)

        # rgb_camXs = d['rgbs']
        # # move channel dim inward, like pytorch wants
        # rgb_camXs = np.transpose(rgb_camXs, axes=[0, 3, 1, 2])
        # rgb_camXs = utils.py.preprocess_color(rgb_camXs)
        # d['rgb_camXs'] = rgb_camXs
        d['filename'] = filename

        if False:
            # this does not work 

            print('filename', filename)
            scene_id = filename.split('.')[-2]
            print('scene_id', scene_id)

            raw_data_dir = "/projects/katefgroup/datasets/mcs/%s_eval3train" % self.data_type_full
            scene_npz_path = '%s/%s.npz' % (raw_data_dir, scene_id)
            d = dict(np.load(scene_npz_path, allow_pickle=True))
            json_dict = d['json_dict']
            json_dict = json_dict.item()
            ans = json_dict['answer']['choice']
            # sem_seq = d['sem_seq']
            # depth_seq = d['depth_seq']
            print('ans', ans)

        
        return d

    def __len__(self):
        return len(self.records)
        # return 10

    def get_item_names(self):
        item_names = [
            'pix_T_cams',
            'rgbs',
            'depths',
            'boxlists',
            'validlists',
            'masks',
        ]
        return item_names

class McsDataset(torch.utils.data.Dataset):
    def __init__(self, data_type='st', shuffle=True, seqlen=50):

        mcs_data_mod = 'ak'
        if data_type=='st':
            data_seqlen = 90
            self.data_type_full = 'spatio_temporal'
        elif data_type=='sc':
            data_seqlen = 60
            self.data_type_full = 'shape_constancy'
        elif data_type=='op':
            data_seqlen = 60
            self.data_type_full = 'object_permanence'
        elif data_type=='all':
            data_seqlen = 60
            self.data_type_full = 'all'
        else:
            assert(False) # not set up yet

        mcs_dataset_location = "/projects/katefgroup/datasets/mcs/processed"

        trainset = "t%s_%s%dt" % (data_type, mcs_data_mod, data_seqlen)
        dataset_location = "%s" % mcs_dataset_location
        dataset_path = '%s/%s.txt' % (dataset_location, trainset)

        print('dataset_path = %s' % dataset_path)
        with open(dataset_path) as f:
            content = f.readlines()
        dataset_location = dataset_path.split('/')[:-1]
        dataset_location = '/'.join(dataset_location)
        print('dataset_loc = %s' % dataset_location)
        
        records = [dataset_location + '/' + line.strip() for line in content]
        nRecords = len(records)
        print('found %d records in %s' % (nRecords, dataset_path))
        nCheck = np.min([nRecords, 1000])
        for record in records[:nCheck]:
            assert os.path.isfile(record), 'Record at %s was not found' % record
        print('checked the first %d, and they seem to be real files' % (nCheck))
 
        self.records = records
        self.shuffle = shuffle
        self.seqlen = seqlen

    def __getitem__(self, index):
        filename = self.records[index]
        d = np.load(filename, allow_pickle=True)
        d = dict(d)

        if self.shuffle:
            d = random_select_single(self.get_item_names(), d, num_samples=self.seqlen)
        else:
            d = non_random_select_single(self.get_item_names(), d, num_samples=self.seqlen)

        return_dict = {}

        pix_T_camXs = torch.from_numpy(d['pix_T_cams']).cuda().float()

        rgb_camXs = d['rgbs']
        # move channel dim inward, like pytorch wants
        rgb_camXs = torch.from_numpy(rgb_camXs).cuda()
        rgb_camXs = utils.improc.preprocess_color(rgb_camXs)

        depth_camXs = torch.from_numpy(d['depths']).cuda().float()
        xyz_camXs = utils.geom.depth2pointcloud(depth_camXs, pix_T_camXs)

        boxlist_camXs = torch.from_numpy(d['boxlists']).cuda().float()
        lrtlist_camXs = utils.geom.convert_boxlist_to_lrtlist(boxlist_camXs)

        scorelist_s = torch.from_numpy(d['validlists']).cuda().float()

        mask_camXs = torch.from_numpy(d['masks']).cuda().float()

        return_dict['rgb_camXs'] = rgb_camXs
        return_dict['depth_camXs'] = depth_camXs
        return_dict['xyz_camXs'] = xyz_camXs
        return_dict['pix_T_camXs'] = pix_T_camXs
        return_dict['lrtlist_camXs'] = lrtlist_camXs
        return_dict['scorelist_s'] = scorelist_s
        return_dict['mask_camXs'] = mask_camXs
        return_dict['filename'] = filename

        if False:
            # this does not work 

            print('filename', filename)
            scene_id = filename.split('.')[-2]
            print('scene_id', scene_id)

            raw_data_dir = "/projects/katefgroup/datasets/mcs/%s_eval3train" % self.data_type_full
            scene_npz_path = '%s/%s.npz' % (raw_data_dir, scene_id)
            d = dict(np.load(scene_npz_path, allow_pickle=True))
            json_dict = d['json_dict']
            json_dict = json_dict.item()
            ans = json_dict['answer']['choice']
            # sem_seq = d['sem_seq']
            # depth_seq = d['depth_seq']
            print('ans', ans)

        return return_dict

    def __len__(self):
        return len(self.records)
        # return 10

    def get_item_names(self):
        item_names = [
            'pix_T_cams',
            'rgbs',
            'depths',
            'boxlists',
            'validlists',
            'masks',
        ]
        return item_names

class McsMaskDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transform=None):

        records = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if file.endswith('.npz')]
        
        nRecords = len(records)
        print('found %d records in %s' % (nRecords, dataset_path))
        nCheck = np.min([nRecords, 1000])
        for record in records[:nCheck]:
            assert os.path.isfile(record), 'Record at %s was not found' % record
        print('checked the first %d, and they seem to be real files' % (nCheck))
 
        self.records = records
        self.transform = transform

    def __getitem__(self, index):
        filename = self.records[index]
        d = np.load(filename, allow_pickle=True)
        d = dict(d)

        x = torch.from_numpy(d['mask']) # 1 x H x W

        if self.transform is not None:
            x = self.transform(x)

        return {'x': x, 'filename': filename}

    def __len__(self):
        return len(self.records)
        # return 10

    def get_item_names(self):
        item_names = [
            'filename',
            'x']
        return item_names

class CaterDataset(torch.utils.data.Dataset):
    def __init__(self, shuffle=True, seqlen=50, env=None):

        #cater_data_mod = 'a'
        #cater_data_seqlen = 300

        cater_data_mod = 'aj'
        cater_data_seqlen = 6

        cater_dataset_location = "/projects/katefgroup/datasets/cater/npzs"

        if (env != 'val') and (env != 'test'):
            set_suffix = "t"
        else:
            set_suffix = "v"


        # trainset = "t%ss%st" % (cater_data_mod, cater_data_seqlen)
        trainset = "m%ss%s%s" % (cater_data_mod, cater_data_seqlen, set_suffix)
        dataset_location = "%s" % cater_dataset_location
        dataset_path = '%s/%s.txt' % (dataset_location, trainset)

        print('dataset_path = %s' % dataset_path)
        with open(dataset_path) as f:
            content = f.readlines()
        dataset_location = dataset_path.split('/')[:-1]
        dataset_location = '/'.join(dataset_location)
        print('dataset_loc = %s' % dataset_location)
        
        records = [dataset_location + '/' + line.strip() for line in content]
        nRecords = len(records)
        print('found %d records in %s' % (nRecords, dataset_path))
        nCheck = np.min([nRecords, 1000])
        for record in records[:nCheck]:
            assert os.path.isfile(record), 'Record at %s was not found' % record
        print('checked the first %d, and they seem to be real files' % (nCheck))
 
        self.records = records
        self.shuffle = shuffle
        self.seqlen = seqlen

    def __getitem__(self, index):
        filename = self.records[index]
        d = np.load(filename, allow_pickle=True)
        d = dict(d)

        if self.shuffle:
            d = random_select_single(self.get_item_names(), d, num_samples=self.seqlen)
        else:
            d = non_random_select_single(self.get_item_names(), d, num_samples=self.seqlen)

        return_dict = {}

        pix_T_camXs = torch.from_numpy(d['pix_T_camXs']).float()

        rgb_camXs = d['rgb_camXs']
        # move channel dim inward, like pytorch wants
        rgb_camXs = np.transpose(rgb_camXs, axes=[0, 3, 1, 2])
        rgb_camXs = utils.py.preprocess_color(rgb_camXs)
        rgb_camXs = torch.from_numpy(rgb_camXs).float()

        xyz_camXs = torch.from_numpy(d['xyz_camXs']).float()

        lrtlist = torch.from_numpy(d['lrt_traj_world']).float()
        world_T_camXs = torch.from_numpy(d['world_T_camXs']).float()
        world_T_camR = torch.from_numpy(d['world_T_camR']).float()
        camXs_T_world = utils.geom.safe_inverse(world_T_camXs)
        lrtlist_camXs = utils.geom.apply_4x4_to_lrtlist(camXs_T_world, lrtlist)

        scorelist_s = torch.from_numpy(d['scorelist']).float()

        return_dict['rgb_camXs'] = rgb_camXs
        return_dict['xyz_camXs'] = xyz_camXs
        return_dict['pix_T_camXs'] = pix_T_camXs
        return_dict['lrtlist_camXs'] = lrtlist_camXs
        return_dict['scorelist_s'] = scorelist_s
        return_dict['world_T_camXs'] = world_T_camXs
        return_dict['world_T_camR'] = world_T_camR
        return_dict['filename'] = filename
        return return_dict

    def __len__(self):
        return len(self.records)
        # return 10

    def get_item_names(self):
        item_names = [
            'pix_T_camXs',
            'rgb_camXs',
            'xyz_camXs',
            'world_T_camXs',
            'lrt_traj_world',
            'scorelist',
            'world_T_camR',
        ]
        return item_names

class CaterCacheDataset(torch.utils.data.Dataset):
    def __init__(self, shuffle=True, seqlen=50):

        cater_data_mod = 'ai'
        cater_data_seqlen = 300

        cater_cache_mod = 'aa'
        cater_cache_location = "./cater_cache"
        self.cache_npzs = glob.glob(os.path.join(cater_cache_location, "*%s.npz" % cater_cache_mod))
        print('found %d cache npzs with mod %s in %s' % (len(self.cache_npzs), cater_cache_mod, cater_cache_location))
        
        # cater_dataset_location = "/projects/katefgroup/datasets/cater/npzs"

        # trainset = "t%ss%st" % (cater_data_mod, cater_data_seqlen)
        # dataset_location = "%s" % cater_dataset_location
        # dataset_path = '%s/%s.txt' % (dataset_location, trainset)

        # print('dataset_path = %s' % dataset_path)
        # with open(dataset_path) as f:
        #     content = f.readlines()
        # dataset_location = dataset_path.split('/')[:-1]
        # dataset_location = '/'.join(dataset_location)
        # print('dataset_loc = %s' % dataset_location)
        
        # records = [dataset_location + '/' + line.strip() for line in content]
        # nRecords = len(records)
        # print('found %d records in %s' % (nRecords, dataset_path))
        # nCheck = np.min([nRecords, 1000])
        # for record in records[:nCheck]:
        #     assert os.path.isfile(record), 'Record at %s was not found' % record
        # print('checked the first %d, and they seem to be real files' % (nCheck))
 
        # self.records = records
        self.shuffle = shuffle
        self.seqlen = seqlen

    def __getitem__(self, index):
        filename = self.cache_npzs[index]
        d = np.load(filename, allow_pickle=True)
        d = dict(d)

        # obj_mem0_e = d['obj_mem0_e']
        # bkg_mem0_e = d['bkg_mem0_e']
        obj_xyz_cam0 = d['obj_xyz_cam0']
        bkg_xyz_cam0 = d['bkg_xyz_cam0']

        filename = np.array_str(d['filename'])

        d = np.load(filename, allow_pickle=True)
        d = dict(d)
        d = non_random_select_single(self.get_item_names(), d, num_samples=self.seqlen)

        rgb_camXs = d['rgb_camXs']
        # move channel dim inward, like pytorch wants
        rgb_camXs = np.transpose(rgb_camXs, axes=[0, 3, 1, 2])
        rgb_camXs = utils.py.preprocess_color(rgb_camXs)
        d['rgb_camXs'] = rgb_camXs
        d['filename'] = filename
        d['obj_xyz_cam0'] = obj_xyz_cam0
        d['bkg_xyz_cam0'] = bkg_xyz_cam0
        return d

    def __len__(self):
        return len(self.cache_npzs)
        # return 10

    def get_item_names(self):
        item_names = [
            'pix_T_camXs',
            'rgb_camXs',
            'xyz_camXs',
            'world_T_camXs',
            'lrt_traj_world',
            'scorelist',
            'world_T_camR',
        ]
        return item_names

class KittiDataset(torch.utils.data.Dataset):
    def __init__(self, shuffle=True, seqlen=50):

        kitti_data_mod = 'ah'
        kitti_data_seqlen = 2
        kitti_data_incr = 1

        kitti_dataset_location = "/projects/katefgroup/datasets/kitti/processed/npzs"

        dataset_name = "ktrack"
        # trainset = "t%ss%si%sa" % (kitti_data_mod, kitti_data_seqlen, kitti_data_incr)
        # trainset = "t%ss%si%sseq11" % (kitti_data_mod, kitti_data_seqlen, kitti_data_incr)
        trainset = "t%ss%si%sseq11" % (kitti_data_mod, kitti_data_seqlen, kitti_data_incr)
        trainset_format = "ktrack"
        trainset_consec = False
        dataset_location = "%s" % kitti_dataset_location

        dataset_path = '%s/%s.txt' % (dataset_location, trainset)

        print('dataset_path = %s' % dataset_path)
        with open(dataset_path) as f:
            content = f.readlines()
        dataset_location = dataset_path.split('/')[:-1]
        dataset_location = '/'.join(dataset_location)
        print('dataset_loc = %s' % dataset_location)

        records = [dataset_location + '/' + line.strip() for line in content]
        nRecords = len(records)
        print('found %d records in %s' % (nRecords, dataset_path))
        nCheck = np.min([nRecords, 1000])
        for record in records[:nCheck]:
            assert os.path.isfile(record), 'Record at %s was not found' % record
        print('checked the first %d, and they seem to be real files' % (nCheck))
 
        self.records = records
        self.shuffle = shuffle
        self.seqlen = seqlen

    def __getitem__(self, index):
        filename = self.records[index]
        d = np.load(filename, allow_pickle=True)
        d = dict(d)

        if self.shuffle:
            d = random_select_single(self.get_item_names(), d, num_samples=self.seqlen)
        else:
            d = non_random_select_single(self.get_item_names(), d, num_samples=self.seqlen)

        rgb_camXs = d['rgb_camXs']
        # move channel dim inward, like pytorch wants
        rgb_camXs = np.transpose(rgb_camXs, axes=[0, 3, 1, 2])
        rgb_camXs = utils.py.preprocess_color(rgb_camXs)
        d['rgb_camXs'] = rgb_camXs
        d['filename'] = filename
        return d

    def __len__(self):
        return len(self.records)
        # return 10

    def get_item_names(self):
        item_names = [
            'rgb_camXs',
            'xyz_veloXs',
            'origin_T_camXs',
            'pix_T_cams',
            'cams_T_velos',
            'boxlists',
            'tidlists',
            'scorelists',
        ]
        return item_names

class KittiCacheDataset(torch.utils.data.Dataset):
    def __init__(self, shuffle=True, seqlen=50):

        kitti_data_mod = 'ah'
        kitti_data_seqlen = 2
        kitti_data_incr = 1

        kitti_cache_mod = 'af'
        kitti_cache_location = "./kitti_cache"
        self.cache_npzs = glob.glob(os.path.join(kitti_cache_location, "*%s.npz" % kitti_cache_mod))
        # self.cache_npzs = glob.glob(os.path.join(kitti_cache_location, "*seq_0011*%s.npz" % kitti_cache_mod))
        print('found %d cache npzs with mod %s in %s' % (len(self.cache_npzs), kitti_cache_mod, kitti_cache_location))
        
        # kitti_dataset_location = "/projects/katefgroup/datasets/kitti/processed/npzs"

        # dataset_name = "ktrack"
        # # trainset = "t%ss%si%st" % (kitti_data_mod, kitti_data_seqlen, kitti_data_incr)
        # trainset = "t%ss%si%sseq11" % (kitti_data_mod, kitti_data_seqlen, kitti_data_incr)
        # trainset_format = "ktrack"
        # trainset_consec = False
        # dataset_location = "%s" % kitti_dataset_location

        # dataset_path = '%s/%s.txt' % (dataset_location, trainset)

        # print('dataset_path = %s' % dataset_path)
        # with open(dataset_path) as f:
        #     content = f.readlines()
        # dataset_location = dataset_path.split('/')[:-1]
        # dataset_location = '/'.join(dataset_location)
        # print('dataset_loc = %s' % dataset_location)

        # records = [dataset_location + '/' + line.strip() for line in content]
        # nRecords = len(records)
        # print('found %d records in %s' % (nRecords, dataset_path))
        # nCheck = np.min([nRecords, 1000])
        # for record in records[:nCheck]:
        #     assert os.path.isfile(record), 'Record at %s was not found' % record
        # print('checked the first %d, and they seem to be real files' % (nCheck))
 
        # self.records = records
        
        self.shuffle = shuffle
        self.seqlen = seqlen

    def __getitem__(self, index):
        filename = self.cache_npzs[index]
        d = np.load(filename, allow_pickle=True)
        d = dict(d)

        # obj_mem0_e = d['obj_mem0_e']
        # bkg_mem0_e = d['bkg_mem0_e']
        obj_xyz_cam0 = d['obj_xyz_cam0']
        bkg_xyz_cam0 = d['bkg_xyz_cam0']

        filename = np.array_str(d['filename'])

        d = np.load(filename, allow_pickle=True)
        d = dict(d)
        d = non_random_select_single(self.get_item_names(), d, num_samples=self.seqlen)

        rgb_camXs = d['rgb_camXs']
        # move channel dim inward, like pytorch wants
        rgb_camXs = np.transpose(rgb_camXs, axes=[0, 3, 1, 2])
        rgb_camXs = utils.py.preprocess_color(rgb_camXs)
        d['rgb_camXs'] = rgb_camXs
        d['filename'] = filename
        # d['obj_mem0_e'] = obj_mem0_e
        # d['bkg_mem0_e'] = bkg_mem0_e
        d['obj_xyz_cam0'] = obj_xyz_cam0
        d['bkg_xyz_cam0'] = bkg_xyz_cam0
        return d

    def __len__(self):
        # return len(self.cache_npzs)
        return 1

    def get_item_names(self):
        item_names = [
            'rgb_camXs',
            'xyz_veloXs',
            'origin_T_camXs',
            'pix_T_cams',
            'cams_T_velos',
            'boxlists',
            'tidlists',
            'scorelists',
        ]
        return item_names


class ArrowDataset(torch.utils.data.Dataset):
    def __init__(self, shuffle=True, seqlen=1):
        # this is an image dataset
        assert (seqlen == 1)
        arrow_dataset_location = "/projects/katefgroup/datasets/arrow/point-out-the-wrong-guy-split"

        print('dataset_path = %s' % arrow_dataset_location)

        records = []
        for image_dir_name in os.listdir(arrow_dataset_location):
            for image_name in os.listdir(os.path.join(arrow_dataset_location, image_dir_name, 'images')):
                records.append(os.path.join(arrow_dataset_location, image_dir_name, "%s", image_name))

        nRecords = len(records)
        print('found %d records in %s' % (nRecords, arrow_dataset_location))

        self.records = records
        self.shuffle = shuffle
        self.seqlen = seqlen

        self.pix_T_cam = np.array([
            [350, 0, 160, 0],
            [0, 350, 120, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)

    def __getitem__(self, index):
        filename = self.records[index]
        image_filename = filename % "images"
        scene_filename = (filename % "scenes").replace('png', 'json')

        rgb_camXs = np.asarray(Image.open(image_filename).convert('RGB'))
        rgb_camXs = rgb_camXs.transpose(2, 0, 1)[np.newaxis, :, :, :]
        rgb_camXs = utils.py.preprocess_color(rgb_camXs)
        rgb_camXs = torch.tensor(rgb_camXs).cuda().float()
        print(rgb_camXs.shape)

        scene_info = json.load(open(scene_filename))
        objs = scene_info['objects']
        pix_coors = []
        for obj in objs:
            pix_coors.append(obj['pixel_coords'])
        pix_coors = torch.tensor(pix_coors).cuda().float()

        return_dict = {}

        return_dict['rgb_camXs'] = rgb_camXs
        return_dict['centers_xyd'] = pix_coors
        return_dict['filename'] = filename
        return return_dict

    def __len__(self):
        return len(self.records)
        # return 10

    def get_item_names(self):
        item_names = [
            'pix_T_camXs',
            'rgb_camXs',
            'xyz_camXs',
            'world_T_camXs',
            'lrt_traj_world',
            'scorelist',
            'world_T_camR',
        ]
        return item_names


class ArrowObjCropDataset(torch.utils.data.Dataset):
    def __init__(self, image_location, dataset_location, crop_size=32, mod='ab', num_videos=-1, do_masking=True,
                 padding=0):
        records = [filename for filename in sorted(os.listdir(dataset_location)) if mod in filename]
        records = sorted(records)

        img_records = []
        for img_dir in os.listdir(image_location):
            for img_filename in os.listdir(os.path.join(image_location, img_dir, 'images')):
                img_records.append(os.path.join(img_dir, 'images', img_filename))
        img_records = sorted(img_records)

        if num_videos > 0:
            records = records[:min(num_videos, len(records))]  # debug
            img_records = img_records[:min(num_videos, len(img_records))]

        nRecords = len(records)
        print('found %d records in %s' % (nRecords, dataset_location))

        nCheck = np.min([nRecords, 1000])
        for record in records[:nCheck]:
            assert os.path.isfile(dataset_location + '/' + record)
        print('checked the first %d, and they seem to be real files' % (nCheck))

        self.records = records
        self.img_records = img_records
        self.dataset_location = dataset_location
        self.image_location = image_location
        self.crop_size = crop_size
        self.mod = mod

        # all_rgbs = dict()
        all_rgbs = []
        all_masks = []

        # flatten everything
        print('loading anns and rgbs into memory...')
        for i, (record, img_record) in enumerate(zip(self.records, self.img_records)):
            rgb = np.asarray(Image.open(os.path.join(self.image_location, img_record)).convert('RGB'))
            rgb = rgb[np.newaxis, :, :, :]
            # rgb = dict(np.load(os.path.join(self.image_location, raw_data_filename), allow_pickle=True))[
            #     'rgb_camXs']  # S x H x W x 3

            # all_rgbs[raw_data_filename] = rgb # S x H x W x 3

            data = np.load(os.path.join(dataset_location, record), allow_pickle=True)
            data = dict(data)
            boxlist_2d = data['boxlist_2d']
            scorelist = data['scorelist']
            seglist = data['seglist']

            rgb = np.transpose(rgb, (0, 3, 1, 2))  # S x 3 x H x W
            rgb = utils.improc.preprocess_color(rgb)
            rgb = torch.from_numpy(rgb)

            _, _, H, W = rgb.shape

            # print('scorelist', scorelist)

            S, N, _ = boxlist_2d.shape
            for n in range(N):
                valid = scorelist[:, n] > 0  # S
                if np.sum(valid) > 0:
                    frame_ids = (np.where(valid)[0])

                    full_mask = np.zeros((np.sum(valid), 1, H, W))

                    for j, frame_id in enumerate(frame_ids):
                        (tl_x, tl_y, region_w, region_h), rle = seglist[frame_id][n]
                        cropped_mask = self.rle_to_mask(rle, region_w, region_h)
                        full_mask[j, 0, tl_y:tl_y + region_h, tl_x:tl_x + region_w] = cropped_mask

                    rgb_obj = rgb[valid]  # s_valid x 3 x H x W
                    box = torch.from_numpy(boxlist_2d[valid, n,])  # s_valid x 4
                    x1, y1, x2, y2 = torch.unbind(box, dim=1)
                    box = torch.stack([y1 - padding, x1 - padding, y2 + padding, x2 + padding], dim=1)

                    full_mask = torch.from_numpy(full_mask).float()

                    # do cropping
                    crops = utils.geom.crop_and_resize(rgb_obj, box, self.crop_size, self.crop_size,
                                                       box2d_is_normalized=False)
                    masks = utils.geom.crop_and_resize(full_mask, box, self.crop_size, self.crop_size,
                                                       box2d_is_normalized=False)

                    all_rgbs.append(crops)
                    all_masks.append(masks)

            if (i + 1) % (len(self.records) // 10) == 0:  # this is slow, so we add a progress bar
                print('finished %d/%d' % (i + 1, len(self.records)))

        self.all_rgbs = torch.cat(all_rgbs)
        self.all_masks = torch.cat(all_masks)

        self.num_crops = len(self.all_rgbs)

        print('found %d crops' % self.num_crops)

    def __getitem__(self, index):
        # read detections
        cropA = self.all_rgbs[index]
        maskA = self.all_masks[index]

        return {'rgb': cropA, 'mask': maskA, 'idx': index}

    def rle_to_mask(self, rle, width, height):
        """
        rle: input rle mask encoding
        each evenly-indexed element represents number of consecutive 0s
        each oddly indexed element represents number of consecutive 1s
        width and height are dimensions of the mask
        output: 2-D binary mask
        """
        # allocate list of zeros
        v = [0] * (width * height)

        # set id of the last different element to the beginning of the vector
        idx_ = 0
        for i in range(len(rle)):
            if i % 2 != 0:
                # write as many 1s as RLE says (zeros are already in the vector)
                for j in range(rle[i]):
                    v[idx_ + j] = 1
            idx_ += rle[i]

        # reshape vector into 2-D mask
        # return np.reshape(np.array(v, dtype=np.uint8), (height, width)) # numba bug / not supporting np.reshape
        return np.array(v, dtype=np.float32).reshape((height, width))

    def __len__(self):
        return self.num_crops


class ArrowBoxDataset(torch.utils.data.Dataset):
    def __init__(self, image_location, dataset_location, crop_size=32, mod='aa', num_videos=-1, use_cache=False):
        cache_file = 'all_boxes_%s.npy' % mod

        if use_cache and os.path.exists(cache_file):
            print("loading data from cache %s" % cache_file)
            records = np.load(cache_file)
        else:
            print("loading data...")
            records = []
            from tqdm import tqdm
            for filename in tqdm(sorted(os.listdir(dataset_location))):
                if mod in filename:
                    det_fullpath = os.path.join(dataset_location, filename)
                    det_meta = np.load(det_fullpath, allow_pickle=True)
                    det_scores = det_meta['scorelist'][0]
                    det_boxes = det_meta['boxlist_2d'][0]
                    records.append(det_boxes[det_scores > 0])
            records = np.concatenate(records, axis=0)
            np.save(cache_file, records)
            print("save cache to %s" % cache_file)

        if num_videos > 0:
            records = records[:min(num_videos, len(records))]  # debug

        nRecords = len(records)
        print('found %d records in %s' % (nRecords, dataset_location))

        # get H and W
        img_records = []
        for img_dir in os.listdir(image_location):
            for img_filename in os.listdir(os.path.join(image_location, img_dir, 'images')):
                img_records.append(os.path.join(img_dir, 'images', img_filename))
        img_records = sorted(img_records)
        rgb = np.asarray(Image.open(os.path.join(image_location, img_records[0])).convert('RGB'))
        self.H, self.W, _ = rgb.shape

        self.records = records
        self.dataset_location = dataset_location
        self.image_location = image_location
        self.crop_size = crop_size
        self.mod = mod

    def __getitem__(self, index):
        boxA = torch.from_numpy(self.records[index]).unsqueeze(0)  # (1, 4)
        c = (boxA[:, :2] + boxA[:, 2:]) / 2
        l = torch.abs(boxA[:, 2:] - boxA[:, :2])
        c[:, 0] /= self.W
        c[:, 1] /= self.H
        l[:, 0] /= self.W
        l[:, 1] /= self.H
        return torch.cat([c, l], dim=1)

    def __len__(self):
        return len(self.records)


class ArrowBackgroundDataset(torch.utils.data.Dataset):
    def __init__(self, image_location, dataset_location, crop_size=64, mod='ab', num_videos=10):
        records = [filename for filename in sorted(os.listdir(dataset_location)) if mod in filename]
        records = sorted(records)

        img_records = []
        for img_dir in os.listdir(image_location):
            for img_filename in os.listdir(os.path.join(image_location, img_dir, 'images')):
                img_records.append(os.path.join(img_dir, 'images', img_filename))
        img_records = sorted(img_records)

        if num_videos > 0:
            records = records[:min(num_videos, len(records))]  # debug
            img_records = img_records[:min(num_videos, len(img_records))]

        nRecords = len(records)
        print('found %d records in %s' % (nRecords, dataset_location))

        nCheck = np.min([nRecords, 1000])
        for record in records[:nCheck]:
            assert os.path.isfile(dataset_location + '/' + record)
        print('checked the first %d, and they seem to be real files' % (nCheck))

        self.records = records
        self.img_records = img_records
        self.dataset_location = dataset_location
        self.image_location = image_location
        self.crop_size = crop_size
        self.mod = mod

    def __getitem__(self, index):
        # read detections
        det_filename = self.records[index]
        det_fullpath = os.path.join(self.dataset_location, det_filename)
        det_meta = np.load(det_fullpath, allow_pickle=True)
        det_scores = det_meta['scorelist']
        det_boxes = det_meta['boxlist_2d']
        seglist = det_meta['seglist']
        det_scores_binary = det_scores > 0

        # read images
        img_filename = self.img_records[index]
        rgb = np.asarray(Image.open(os.path.join(self.image_location, img_filename)).convert('RGB'))
        rgb = rgb[np.newaxis, :, :, :]
        rgb = np.transpose(rgb, (0, 3, 1, 2))  # S x 3 x H x W
        rgb = utils.improc.preprocess_color(rgb)
        rgb = torch.from_numpy(rgb).float()
        _, _, H, W = rgb.shape
        S, N = det_scores_binary.shape

        full_mask = np.zeros((N, 1, H, W))

        timestep = 0
        for obj_id in range(N):
            if det_scores_binary[timestep, obj_id]:
                (tl_x, tl_y, region_w, region_h), rle = seglist[timestep][obj_id]
                cropped_mask = self.rle_to_mask(rle, region_w, region_h)
                full_mask[obj_id, 0, tl_y:tl_y + region_h, tl_x:tl_x + region_w] = cropped_mask

        full_mask = np.sum(full_mask, axis=0, keepdims=True)
        full_mask = np.clip(full_mask, 0.0, 1.0)
        full_mask = 1.0 - full_mask  # take inverse. 1.0 for background areas

        full_mask = torch.from_numpy(full_mask).float()
        rgb_frame = rgb[timestep].unsqueeze(0)

        full_mask = utils.improc.erode2d(full_mask, times=2, device='cpu')

        full_mask = F.interpolate(full_mask, size=(self.crop_size, self.crop_size)).squeeze(0)
        rgb_frame = F.interpolate(rgb_frame, size=(self.crop_size, self.crop_size)).squeeze(0)

        return {'rgb': rgb_frame, 'mask': full_mask}

    def rle_to_mask(self, rle, width, height):
        """
        rle: input rle mask encoding
        each evenly-indexed element represents number of consecutive 0s
        each oddly indexed element represents number of consecutive 1s
        width and height are dimensions of the mask
        output: 2-D binary mask
        """
        # allocate list of zeros
        v = [0] * (width * height)

        # set id of the last different element to the beginning of the vector
        idx_ = 0
        for i in range(len(rle)):
            if i % 2 != 0:
                # write as many 1s as RLE says (zeros are already in the vector)
                for j in range(rle[i]):
                    v[idx_ + j] = 1
            idx_ += rle[i]

        # reshape vector into 2-D mask
        # return np.reshape(np.array(v, dtype=np.uint8), (height, width)) # numba bug / not supporting np.reshape
        return np.array(v, dtype=np.float32).reshape((height, width))

    def __len__(self):
        return len(self.records)


class CocoCropDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 json_path="/projects/katefgroup/datasets/detectron2_datasets/coco/annotations/instances_train2017.json", 
                 image_path="/projects/katefgroup/datasets/detectron2_datasets/coco/train2017",
                 crop_size=64,
                 padding=0,
                 transform=None):

        self.image_path = image_path
        self.image_template = "%012d.jpg"

        with open(json_path) as f:
            coco_json = json.load(f)

        annotations_all = coco_json['annotations']
        annotations_by_images = {}
        for i in range(len(annotations_all)):
            if annotations_all[i]['iscrowd'] != 0:
                continue
            img_id = annotations_all[i]['image_id']
            if img_id in annotations_by_images:
                annotations_by_images[img_id].append(annotations_all[i])
            else:
                annotations_by_images[img_id] = [annotations_all[i]]

        # randomize and select some good annotations
        annotations_list = list(annotations_by_images.values())
        random.shuffle(annotations_list)
        n_to_include = 1000
        n_included = 0
        n_at = -1
        self.annotations = []
        while n_included < n_to_include:
            n_at += 1
            annotation_list = annotations_list[n_at]
            valid = np.ones((len(annotation_list)))

            img_fn = os.path.join(self.image_path, self.image_template % annotation_list[0]['image_id'])
            rgb = np.array(Image.open(img_fn).convert('RGB')) # in range [0, 255]
            H, W, _ = rgb.shape           

            boxmask = np.zeros((H, W)) - 1
            for i, anno in enumerate(annotation_list):
                bbox = anno['bbox']
                x1, y1, x2, y2 = max(0,bbox[0]-padding), max(0,bbox[1]-padding), min(W, bbox[0]+bbox[2]+padding), min(H,bbox[1]+bbox[3]+padding)
                # overlap check
                in_region_indices = np.unique(boxmask[int(y1):int(y2), int(x1):int(x2)])
                in_region_indices = in_region_indices[in_region_indices!=-1]
                if len(in_region_indices) > 1:
                    valid[in_region_indices.astype(int)] = 0
                    valid[i] = 0
                    boxmask[int(y1):int(y2), int(x1):int(x2)] = i
                    continue
                boxmask[int(y1):int(y2), int(x1):int(x2)] = i
                # touching border check
                if x1 == 0 or y1 == 0 or x2 == W or y2 == H:
                    valid[i] = 0
                    continue
                if y2 - y1 < 50 and x2 - x1 < 50:
                    valid[i] = 0
                    continue
            valid = valid[:-1]
            if valid.sum() == 0:
                continue

            choose_idx = np.random.choice(np.array(np.where(valid == 1)).reshape(-1))
            self.annotations.append(annotation_list[choose_idx]) 
                
            n_included += 1
            print(n_included)

        self.crop_size = crop_size
        self.padding = padding
        self.transform = transform

    def __getitem__(self, index):
        # fetch annotations
        anno = self.annotations[index]
        polygons = anno['segmentation']
        bbox = anno['bbox']

        # read rgb
        img_fn = os.path.join(self.image_path, self.image_template % anno['image_id'])
        rgb = np.array(Image.open(img_fn).convert('RGB')) # in range [0, 255]
        height, width, _ = rgb.shape

        # obtain mask
        mask = polygons_to_bitmask(polygons, height, width)

        # make crops
        rgb = utils.improc.preprocess_color(torch.from_numpy(rgb.astype(float)).permute(2,0,1)).unsqueeze(0).float() # (1,3,H,W)
        mask = torch.from_numpy(mask.astype(float)).unsqueeze(0).unsqueeze(0).float() # (1,1,H,W)
        x1, y1, x2, y2 = bbox[0]-self.padding, bbox[1]-self.padding, bbox[0]+bbox[2]+self.padding, bbox[1]+bbox[3]+self.padding
        bbox_torch = torch.as_tensor([y1, x1, y2, x2]).reshape(1,4).float()
        rgb_crop = utils.geom.crop_and_resize(rgb, bbox_torch, self.crop_size, self.crop_size, box2d_is_normalized=False).squeeze(0) # (3,H,W)
        mask_crop = utils.geom.crop_and_resize(mask, bbox_torch, self.crop_size, self.crop_size, box2d_is_normalized=False).squeeze(0) # (1,H,W)

        '''
        if np.random.rand() < 0.5:
            # horizontal flip
            rgb_crop = torch.flip(rgb_crop, [2])
            mask_crop = torch.flip(mask_crop, [2])
        '''

        return {'rgb': rgb_crop, 'mask': mask_crop, 'index': torch.as_tensor(index)}

    def __len__(self):
        return len(self.annotations)


def random_select_single(item_names, batch, num_samples=2):
    # num_all = len(batch[item_names[origin]]) # total number of frames

    num_all = len(batch[item_names[0]]) # total number of frames
    # print('num_all', num_all)
    
    stride = 1
    # stride = np.random.randint(1, 3)

    start_inds = list(range(num_all-num_samples))
    if num_all==num_samples:
        start_ind = 0
    else:
        start_ind = np.random.randint(0, num_all-num_samples*stride, 1).squeeze()

    batch_new = {}
    inds = list(range(start_ind,start_ind+num_samples*stride,stride))
    for item_name in item_names:
        item = batch[item_name]
        item = item[inds]
        batch_new[item_name] = item

    batch_new['ind_along_S'] = inds
    return batch_new

def non_random_select_single(item_names, batch, num_samples=2):

    num_all = len(batch[item_names[0]]) # total number of frames

    batch_new = {}
    # select valid candidate
    sample_id = range(num_all)

    if len(sample_id) > num_samples:
        # final_sample = sample_id[-num_samples:]
        # final_sample = sample_id[:num_samples]
        # offset = int(num_all/2) # take from the middl
        offset = 0
        final_sample = sample_id[offset:offset+num_samples]
    else:
        final_sample = sample_id

    if num_samples > len(sample_id):
        print('warning: S larger than valid frames number')
        print('num_samples = %d;, len sample = %d;' % (num_samples, len(sample_id)))

    for item_name in item_names:
        item = batch[item_name]
        item = item[final_sample]
        batch_new[item_name] = item

    batch_new['ind_along_S'] = final_sample
    return batch_new
