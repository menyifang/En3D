# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import glob
import json
import torch
import numpy as np
from render import util
from .dataset import Dataset
import cv2

def get_camera_params(resolution= 512, fov=45, elev_angle=-20, azim_angle=0):
    fovy   = np.deg2rad(fov)
    elev = np.radians( elev_angle )
    azim = np.radians( azim_angle )
    proj_mtx = util.perspective(fovy, resolution /resolution, 1, 50)
    mv     = util.translate(0, 0, -3) @ (util.rotate_x(elev) @ util.rotate_y(azim))
    normal_rotate =  util.rotate_y_1(-azim ) @ util.rotate_x_1(-elev)
    # nomral_rotate =  util.rotate_y_1(0) @ util.rotate_x_1(0)
    mvp    = proj_mtx @ mv
    campos = torch.linalg.inv(mv)[:3, 3]
    bkgs = torch.ones(1, resolution, resolution, 3, dtype=torch.float32, device='cuda')
    return {
        'mvp' : mvp[None, ...].cuda(),
        'mv' : mv[None, ...].cuda(),
        'campos' : campos[None, ...].cuda(),
        'resolution' : [resolution, resolution],
        'spp' : 1,
        'background' : bkgs,
        'normal_rotate' : normal_rotate[None,...].cuda(),
        }

def _load_img(path):
    # print(path)
    files = glob.glob(path + '.*')
    assert len(files) > 0, "Tried to find image file for: %s, but found 0 files" % (path)
    img = util.load_image_raw(files[0])
    if img.dtype != np.float32: # LDR image
        img = torch.tensor(img / 255, dtype=torch.float32)
        # img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.tensor(img, dtype=torch.float32)
    return img

class Dataset3D(Dataset):
    def __init__(self, cfg_path, FLAGS, examples=None):
        self.FLAGS = FLAGS
        self.examples = examples
        self.base_dir = os.path.dirname(cfg_path)

        # Load config / transforms
        self.cfg = json.load(open(cfg_path, 'r'))
        self.n_images = len(self.cfg['frames'])

        # Determine resolution & aspect ratio
        print(self.base_dir)
        print(self.cfg['frames'][0]['file_path'])
        self.resolution = _load_img(os.path.join(self.base_dir, self.cfg['frames'][0]['file_path'])).shape[0:2]
        self.aspect = self.resolution[1] / self.resolution[0]

        if self.FLAGS.local_rank == 0:
            print("Dataset3D: %d images with shape [%d, %d]" % (self.n_images, self.resolution[0], self.resolution[1]))

        # Pre-load from disc to avoid slow png parsing
        if self.FLAGS.pre_load:
            self.preloaded_data = []
            for i in range(self.n_images):
                self.preloaded_data += [self._parse_frame(self.cfg, i)]

        self.front_mv = self.load_front_mv(std_view=6)

        print('data init finished!')

    def _parse_frame(self, cfg, idx):
        # Config projection matrix (static, so could be precomputed)
        fovy = util.fovx_to_fovy(cfg['camera_angle_x'], self.aspect)
        proj = util.perspective(fovy, self.aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # Load image data and modelview matrix
        str_to_rep = 'image'
        str_to_use = 'rgba'
        rgba_path = os.path.join(self.base_dir, cfg['frames'][idx]['file_path'])[::-1].replace(
            str_to_rep[::-1], str_to_use[::-1], 1)[::-1]
        # print('rgba_path: ', rgba_path)
        img = _load_img(rgba_path)
        str_to_use = 'normal'
        norm_path = os.path.join(self.base_dir, cfg['frames'][idx]['file_path'])[::-1].replace(
            str_to_rep[::-1], str_to_use[::-1], 1)[::-1]
        normals = _load_img(norm_path)

        mv     = torch.linalg.inv(torch.tensor(cfg['frames'][idx]['transform_matrix'], dtype=torch.float32))
        campos = torch.linalg.inv(mv)[:3, 3]
        mvp    = proj @ mv

        return img[None, ...], normals[None, ...], mv[None, ...], mvp[None, ...], campos[None, ...] # Add batch dimension

    def load_front_mv(self, std_view = 6):
        print('load %s as front view' % self.cfg['frames'][std_view]['file_path'])
        data = self._parse_frame(self.cfg, std_view)
        mv = data[2]
        return mv

    def __len__(self):
        return self.n_images if self.examples is None else self.examples

    def __getitem__(self, itr):

        ### for model_v1
        not_use_view = [2, 10]
        ### for model_0911
        # not_use_view = [3, 9]
        if itr % self.n_images in not_use_view:
            itr = 6

        iter_res = self.FLAGS.train_res
        
        img      = []
        fovy     = util.fovx_to_fovy(self.cfg['camera_angle_x'], self.aspect)

        if self.FLAGS.pre_load:
            # print('pre_load')
            img, norm, mv, mvp, campos = self.preloaded_data[itr % self.n_images]
        else:
            # print('_parse_frame')
            img, norm, mv, mvp, campos = self._parse_frame(self.cfg, itr % self.n_images)
        view_id = torch.tensor(itr % self.n_images)

        out = {
            'mv' : mv,
            'mvp' : mvp,
            'campos' : campos,
            'resolution' : iter_res,
            'spp' : self.FLAGS.spp,
            'img' : img,
            'norm' : norm,
            'front_mv' : self.front_mv,
            'view_id': view_id,
        }

        return out
