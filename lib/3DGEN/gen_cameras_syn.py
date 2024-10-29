# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
import os
from argparse import ArgumentParser
import sys
import cv2 as cv
from glob import glob
import json
from tqdm import tqdm
import cv2
import shutil
import misc  # NOQA
from PIL import Image, ImageFile
import math

def split_image(img, n_row=1, n_col=21):
    # split image into n_row * n_col
    h, w, c = img.shape
    assert h % n_row == 0
    assert w % n_col == 0
    h_per_row = h // n_row
    w_per_col = w // n_col
    imgs = []
    for i in range(n_row):
        for j in range(n_col):
            imgs.append(img[i*h_per_row:(i+1)*h_per_row, j*w_per_col:(j+1)*w_per_col, :])
    return imgs

def flip_yaw(pose_matrix):
    flipped = pose_matrix.copy()
    flipped[0, 1] *= -1
    flipped[0, 2] *= -1
    flipped[1, 0] *= -1
    flipped[2, 0] *= -1
    flipped[0, 3] *= -1
    return flipped

def load_K_Rt_from_P(filename, P=None):
    # This function is borrowed from IDR: https://github.com/lioryariv/idr
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

class Converter(object):
    def __init__(self):
        # self.data_dir = args.data_dir
        # self.seed = args.seed
        # self.out_dir = args.out_dir
        self.type = 'human'

        seg_path = '/data/qingyao/models/matting_human.pb'
        if not os.path.exists(seg_path):
            seg_path = '/data/qingyao/neuralRendering/mycode/service/human_nerf/assets/matting_human.pb'
        # self.segmenter = human_segmenter(model_path=seg_path)

    def dtu_to_json(self, scene_path):

        out = {
            "k1": 0.0,  # take undistorted images only
            "k2": 0.0,
            "k3": 0.0,
            "k4": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "is_fisheye": False,
            "frames": []
        }

        camera_param = dict(np.load(os.path.join(scene_path, 'cameras_sphere.npz')))
        images_lis = sorted(glob(os.path.join(scene_path, 'image/*.png')))
        for idx, image in enumerate(images_lis):
            image = os.path.basename(image)

            world_mat = camera_param['world_mat_%d' % idx]
            scale_mat = camera_param['scale_mat_%d' % idx]

            # scale and decompose
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsic_param, c2w = load_K_Rt_from_P(None, P)
            c2w_gl = misc.cv_to_gl(c2w)

            frame = {"file_path": 'image/' + image.split('.')[0], "transform_matrix": c2w_gl.tolist()}
            out["frames"].append(frame)

            ## PIL to numpy
            # img = Image.open(os.path.join(scene_path, 'image', image))
            # img = np.array(img)[..., ::-1]
            # mask = cv2.imread(os.path.join(scene_path, 'mask', image))[..., 0]
            # rgba = np.concatenate((img, mask[..., None]), axis=-1)
            # os.makedirs(os.path.join(scene_path, 'rgba'), exist_ok=True)
            # cv2.imwrite(os.path.join(scene_path, 'rgba', image), rgba)

        fl_x = intrinsic_param[0][0]
        fl_y = intrinsic_param[1][1]
        cx = intrinsic_param[0][2]
        cy = intrinsic_param[1][2]
        sk_x = intrinsic_param[0][1]
        sk_y = intrinsic_param[1][0]
        img = Image.open(os.path.join(scene_path, 'image', image))
        w, h = img.size

        angle_x = math.atan(w / (fl_x * 2)) * 2
        angle_y = math.atan(h / (fl_y * 2)) * 2

        scale_mat = scale_mat.astype(float)

        out.update({
            "camera_angle_x": angle_x,
            "camera_angle_y": angle_y,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": cx,
            "cy": cy,
            "sk_x": sk_x,
            "sk_y": sk_y,
            "w": int(w),
            "h": int(h),
            "aabb_scale": np.exp2(np.rint(np.log2(scale_mat[0, 0]))),  # power of two, for INGP resolution computation
            "sphere_center": [scale_mat[0, -1], scale_mat[1, -1], scale_mat[2, -1]],
            "sphere_radius": scale_mat[0, 0],
            "centered": True,
            "scaled": True,
        })

        file_path = os.path.join(scene_path, 'transforms_train.json')
        with open(file_path, "w") as outputfile:
            json.dump(out, outputfile, indent=2)
        print('Writing data to json file: ', file_path)
        # copy file to test
        file_path = os.path.join(scene_path, 'transforms_test.json')
        with open(file_path, "w") as outputfile:
            json.dump(out, outputfile, indent=2)
        print('Writing data to json file: ', file_path)

    def convert_data(self, args):
        print(args)

        data_dir = args['data_dir']
        seed = args['seed']
        out_dir = args['out_dir']

        type = self.type

        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'image'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'mask'), exist_ok=True)

        target_img = cv2.imread(os.path.join(data_dir, 'seed%04d.png' % seed))
        if type == 'human':
            n_col = 13
            seed_img_view = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
            use_view = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
            # use_view = [0]
        else:
            n_col = 21
            seed_img_view = [-37, -32, -28, -24, -20, -17, -14, -11, -7, -3, 0, 3, 7, 11, 14, 17, 20, 24, 28, 32, 37]
            use_view = [-37, -32, -24, -20, -17, -14, -11, 0, 11, 14, 17, 20, 24, 32, 37]

        image_list = split_image(target_img, n_row=1, n_col=n_col)
        n_images = len(image_list)

        # load cam_ada
        cam_ada_path = os.path.join(data_dir, 'seed%04d.npy' % seed)
        cam_ada = None
        if os.path.exists(cam_ada_path):
            cam_ada = np.load(cam_ada_path)
            print('cam_ada', cam_ada.shape)


        cam_dict = {}
        print('find {} images'.format(n_images))

        idx = 0
        for view_id in tqdm(use_view):
            print('view_id', view_id)
            cam = cam_ada[seed_img_view.index(view_id)]
            intrinsic_raw = cam[16:].reshape(3, 3)
            pose = cam[:16].reshape(4, 4)

            gt_img_src = os.path.join(os.path.dirname(data_dir), 'target.png')
            gt_img_src_back = os.path.join(os.path.dirname(data_dir), 'target_back.png')

            if view_id == 0 and os.path.exists(gt_img_src):
                img = cv.imread(gt_img_src, -1)
                c = img.shape[2]
                if c==4:
                    color = img[:,:,:3]
                    alpha = img[:,:,3] / 255
                    bk = np.zeros_like(color)
                    color = color * alpha[:, :, np.newaxis] + bk * (1 - alpha[:, :, np.newaxis])
                    color = color.astype(np.uint8)
                    img = color.copy()
            # elif view_id == 6 and os.path.exists(gt_img_src_back):
            #     img = cv.imread(gt_img_src_back)
            else:
                img = image_list[seed_img_view.index(view_id)]
            cv.imwrite(os.path.join(out_dir, 'image', '%04d.png' % idx), img)

            # rgba = self.segmenter.run(img)
            # mask = rgba[:, :, 3]
            # cv.imwrite(os.path.join(out_dir, 'mask', '%04d.png' % idx), mask)

            intrinsic = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
            intrinsic[:3, :3] = intrinsic_raw
            # norm to original
            image_size = 512
            intrinsic[0, 0] = intrinsic[0, 0] * image_size
            intrinsic[1, 1] = intrinsic[1, 1] * image_size
            intrinsic[0, 2] = intrinsic[0, 2] * image_size
            intrinsic[1, 2] = intrinsic[1, 2] * image_size
            # world_mat = intrinsic @ pose
            # world_mat = world_mat.astype(np.float32)

            w2c = np.linalg.inv(pose)
            world_mat = intrinsic @ w2c
            world_mat = world_mat.astype(np.float32)

            cam_dict['camera_mat_{}'.format(idx)] = intrinsic
            cam_dict['camera_mat_inv_{}'.format(idx)] = np.linalg.inv(intrinsic)
            cam_dict['world_mat_{}'.format(idx)] = world_mat
            cam_dict['world_mat_inv_{}'.format(idx)] = np.linalg.inv(world_mat)

            idx += 1

        scale_mat = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
        for i in range(n_images):
            cam_dict['scale_mat_{}'.format(i)] = scale_mat
            cam_dict['scale_mat_inv_{}'.format(i)] = np.linalg.inv(scale_mat)

        np.savez(os.path.join(out_dir, 'cameras_sphere.npz'), **cam_dict)
        self.dtu_to_json(out_dir)

        # copy plane
        plane_src = os.path.join(data_dir, 'seed%04d_triplane.npy' % seed)
        plane_dst = os.path.join(out_dir, 'plane.npy')
        if os.path.exists(plane_src):
            print('copy plane')
            shutil.copy(plane_src, plane_dst)

        # copy obj
        obj_src = os.path.join(data_dir, 'seed%04d.obj' % seed)
        obj_dst = os.path.join(out_dir, 'mesh.obj')
        shutil.copy(obj_src, obj_dst)

        # copy camera
        cam_src = os.path.join(data_dir, 'seed%04d.npy' % seed)
        cam_dst = os.path.join(out_dir, 'camera.npy')
        shutil.copy(cam_src, cam_dst)

        # copy gt front image if exist
        gt_img_src = os.path.join(os.path.dirname(data_dir), 'target.png')
        if os.path.exists(gt_img_src):
            gt_img_dst = os.path.join(out_dir, 'gt.png')
            shutil.copy(gt_img_src, gt_img_dst)

        # copy gt back image if exist
        gt_img_src_back = os.path.join(os.path.dirname(data_dir), 'target_back.png')
        if os.path.exists(gt_img_src_back):
            gt_img_dst = os.path.join(out_dir, 'gt_back.png')
            shutil.copy(gt_img_src_back, gt_img_dst)

        # copy gt side image if exist
        gt_img_src_side = os.path.join(os.path.dirname(data_dir), 'target_side.png')
        if os.path.exists(gt_img_src_side):
            gt_img_dst = os.path.join(out_dir, 'gt_side.png')
            shutil.copy(gt_img_src_side, gt_img_dst)







if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    convert_cameras(args)
