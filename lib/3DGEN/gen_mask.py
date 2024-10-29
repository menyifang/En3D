# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import os
from argparse import ArgumentParser
from glob import glob
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

def human_matting_fn(args):
    data_dir = args.data_dir
    os.makedirs(os.path.join(data_dir, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'rgba'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'color'), exist_ok=True)

    img_path_list = sorted(glob(os.path.join(data_dir, 'image', '*.png')))

    # segment image
    algo_matting = pipeline(Tasks.portrait_matting, model='damo/cv_unet_image-matting')
    for img_path in img_path_list:

        result = algo_matting(img_path)
        rgba = result[OutputKeys.OUTPUT_IMG]

        mask = rgba[:, :, 3]
        outpath = os.path.join(data_dir, 'mask', os.path.basename(img_path))
        cv2.imwrite(outpath, mask)
        print('save mask to %s' % outpath)

        color = rgba[:, :, :3]
        alpha = mask / 255
        bk = np.zeros_like(color)
        color = color * alpha[:, :, np.newaxis] + bk * (1 - alpha[:, :, np.newaxis])
        color = color.astype(np.uint8)
        rgba[:, :, :3] = color
        outpath = os.path.join(data_dir, 'rgba', os.path.basename(img_path))
        cv2.imwrite(outpath, rgba)
        print('save rgba to %s' % outpath)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    args = parser.parse_args()

    human_matting_fn(args)

