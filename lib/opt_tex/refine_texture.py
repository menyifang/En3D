# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import numpy as np
import cv2
import argparse
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import shutil
from segmenter import Segmenter

def initialize_mask(box_hight, box_width, mode='up', maxn=None):
    # modify
    h, w = [box_hight, box_width]
    mask = np.ones((h, w), np.uint8)
    if mode == "up":
        mask[0, :] = 0
    elif mode == "down":
        mask[-1, :] = 0
    elif mode == "left":
        mask[:, 0] = 0
    elif mode == "right":
        mask[:, -1] = 0
    else:
        mask[:, 0] = 0
        mask[:, -1] = 0
        mask[0, :] = 0
        mask[-1, :] = 0

    # mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    if maxn is None:
        maxn = max(w, h) * 0.1

    mask[(mask < 255) & (mask > 0)] = mask[(mask < 255) & (mask > 0)] / maxn
    mask = np.clip(mask, 0, 1)
    return mask.astype(float)


def detect_skin(img, ref_color):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imwrite('hand_skin_ori.png', img)

    lower = ref_color - np.array([20, 50, 50], dtype="uint8")
    upper = ref_color + np.array([20, 50, 50], dtype="uint8")

    mask = cv2.inRange(img, lower, upper)

    return mask

def compute_scale(src, trg):
    h1, w1, _ = src.shape
    h2, w2, _ = trg.shape

    return (h2/h1, w2/w1)

def refine_arm_tex(img, loc, skin_mask, template):

    h_crop, w_crop, rec_h, rec_w = loc['h_crop'], loc['w_crop'], loc['rec_h'], loc['rec_w']
    h_crop = int(h_crop * img.shape[0])
    w_crop = int(w_crop * img.shape[1])
    rec_h = int((rec_h+loc['h_crop']) * img.shape[0])-h_crop
    rec_w = int((rec_w+loc['w_crop']) * img.shape[1])-w_crop

    loc_mask = skin_mask[h_crop:h_crop + rec_h, w_crop:w_crop + rec_w]

    h_pixels, w_pixels = np.where(loc_mask > 0)
    h_crop = max(h_crop + min(h_pixels), h_crop)
    rec_h = int((loc['h_crop'] + loc['rec_h']) * img.shape[0]) - h_crop

    scale = compute_scale(img, template)
    h_crop_new = int(h_crop*scale[0])
    rec_h_new = int(rec_h*scale[0])
    w_crop_new = int(w_crop * scale[1])
    rec_w_new = int(rec_w * scale[1])

    img_hand = template[h_crop_new:h_crop_new + rec_h_new, w_crop_new:w_crop_new + rec_w_new, :]
    img_hand = cv2.resize(img_hand, (rec_w, rec_h), interpolation=cv2.INTER_AREA)

    mask_region = initialize_mask(img_hand.shape[0], img_hand.shape[1], mode='up')
    mask_region = np.expand_dims(mask_region, 2)
    img[h_crop:h_crop + rec_h, w_crop:w_crop + rec_w, :] = img_hand * mask_region + img[h_crop:h_crop + rec_h,
                                                                                    w_crop:w_crop + rec_w, :] * (
                                                                       1 - mask_region)

    return img


class TexRefiner(object):

    def __init__(self, asset_dir='./assets', mode=None):
        print('TexRefiner init done.')
        self.template_path = os.path.join(asset_dir, 'body_tex4.png')
        self.skin_mask_path = os.path.join(asset_dir, 'skin_mask.png')
        self.algo_inpainting = pipeline(Tasks.image_inpainting, model='damo/cv_fft_inpainting_lama')
        self.algo_skin_seg = Segmenter(model_path=os.path.join(asset_dir, 'matting_skin.pb'))

        print(mode)
        if mode is not None and mode=='seed':
            print('use seed skin_loc')
            self.skin_loc = {'h_crop': 0.05, 'w_crop': 1 / 2, 'rec_h': 1 / 100, 'rec_w': 1 / 100}
            # self.skin_loc = {'h_crop': 0.05, 'w_crop': 0.45, 'rec_h': 1 / 100, 'rec_w': 1 / 100}
        else:
            print('use pti skin_loc')
            self.skin_loc = {'h_crop': 0.15, 'w_crop': 0.415, 'rec_h': 1 / 100, 'rec_w': 1 / 100}

    def parser_arm_postion(self, skin_mask, loc, arm_top_range):
        upper = arm_top_range['upper']
        lower = arm_top_range['lower']
        h_crop, w_crop, rec_h, rec_w = loc['h_crop'], loc['w_crop'], loc['rec_h'], loc['rec_w']
        h_crop = int(h_crop * skin_mask.shape[0])
        w_crop = int(w_crop * skin_mask.shape[1])
        rec_h = int((rec_h + loc['h_crop']) * skin_mask.shape[0]) - h_crop
        rec_w = int((rec_w + loc['w_crop']) * skin_mask.shape[1]) - w_crop

        loc_mask = skin_mask[h_crop:h_crop + rec_h, w_crop:w_crop + rec_w]
        loc_cood = loc_mask>0
        loc_cood = np.sum(loc_cood, axis=1)

        loc_cood[loc_cood<20] = 0
        pixels = np.where(loc_cood>0)[0][::-1]

        arm_top = pixels[0]
        for i in range(len(pixels)):
            if i==0:
                continue
            if pixels[i] >= upper:
                arm_top = pixels[i]
                continue
            if pixels[i] <= lower:
                break
            if pixels[i - 1] - pixels[i] <= 5:
                arm_top = pixels[i]
            else:
                break

        return arm_top

    def refine_arm(self, texture_path):
        img = cv2.imread(texture_path)
        print('recover arm...')

        # generate skin mask of front image
        front_path = os.path.join(os.path.dirname(os.path.dirname(texture_path)), 'gt.png')
        front_img = cv2.imread(front_path)
        front_h, front_w = front_img.shape[:2]
        seg_result = self.algo_skin_seg.run(front_img)
        skin_mask_front = seg_result[:, :, 0]
        # extract arm mask
        left_shoulder = [0.236, 0.4] # (h, w)
        left_shoulder = [int(left_shoulder[0]*front_h), int(left_shoulder[1]*front_w)]
        skin_mask_front[:left_shoulder[0], :] = 0
        skin_mask_front[:, left_shoulder[1]:] = 0
        skin_mask_front[int(front_h*0.7):, :] = 0

        # compute arm skin range
        h_pixels_arm = np.where(skin_mask_front>0)[0]
        h_min, h_max = np.min(h_pixels_arm), np.max(h_pixels_arm)
        arm_skin_len = (h_max - h_min)/150/0.927 # 512 size下

        step = 0.01
        arm_top_range = {'upper': (1-arm_skin_len+step)*1150, 'lower': (1-arm_skin_len-step)*1150}

        # exract arm skin color
        skin_mask_front = skin_mask_front / 255
        arm_skin_color = np.mean(front_img[skin_mask_front > 0], axis=0)
        color_image = np.full((1, 1, 3), arm_skin_color, dtype=np.uint8)
        color_image_hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        arm_skin_color_hsv = np.mean(color_image_hsv, axis=0)
        skin_mask_uv = detect_skin(img, arm_skin_color_hsv)

        # adapt template color to arm color
        template = cv2.imread(self.template_path).astype(np.float32)
        template_mean_rgb = np.array([137.25, 139.44, 172.72], dtype="float32")[None, None]
        template = template + (arm_skin_color[None, None] - template_mean_rgb)
        template = np.clip(template, 0, 255).astype(np.uint8)

        arm_loc = {'h_crop': 0, 'w_crop': 0, 'rec_h': 1 / 2, 'rec_w': 1 / 6}
        left_arm_top = self.parser_arm_postion(skin_mask_uv, arm_loc, arm_top_range)
        print('left_arm_top: ', left_arm_top)
        arm_loc['h_crop'] = left_arm_top/img.shape[0]
        arm_loc['rec_h'] = 1/2 - arm_loc['h_crop']
        img = refine_arm_tex(img, arm_loc, skin_mask_uv, template)

        arm_loc = {'h_crop': 0, 'w_crop': 5 / 6, 'rec_h': 1 / 2, 'rec_w': 1 / 6}
        step = 0.03
        arm_top_range = {'upper': (1-arm_skin_len+step)*1150, 'lower': (1-arm_skin_len-step)*1150}
        left_arm_top = self.parser_arm_postion(skin_mask_uv, arm_loc, arm_top_range)
        arm_loc['h_crop'] = left_arm_top / img.shape[0]
        arm_loc['rec_h'] = 1 / 2 - arm_loc['h_crop']
        img = refine_arm_tex(img, arm_loc, skin_mask_uv, template)

        cv2.imwrite(texture_path, img)


def main(args):
    print('args.mode: ', args.mode)
    inpath = args.inpath
    outpath = args.outpath
    shutil.copy(inpath, outpath)

    ## refine face left-right symmetry
    texref = TexRefiner(asset_dir='../../models', mode=args.mode)

    ## enhance full texture
    portrait_enhancement = pipeline(Tasks.image_portrait_enhancement,
                                        model='damo/cv_gpen_image-portrait-enhancement')
    result = portrait_enhancement(outpath)
    cv2.imwrite(outpath, result[OutputKeys.OUTPUT_IMG])

    ## refine arm texture
    texref.refine_arm(outpath)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', default='', type=str)
    parser.add_argument('--outpath', default='', type=str)
    parser.add_argument('--mode', default='inversion', type=str)

    args = parser.parse_args()

    main(args)
