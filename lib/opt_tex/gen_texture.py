# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import cv2
import utils
import argparse
from uv_unwrapping import Unwrapper
from optimize_tex import Texture

def main(args):
    smpl_path = args.smpl_obj_path
    refine_path = args.refine_path
    data_dir = args.data_dir

    indir = os.path.dirname(refine_path)

    assert smpl_path[-4:] == '.obj'
    assert refine_path[-4:] == '.obj'

    reduced_body_mesh = utils.read_obj(refine_path)
    smpl_mesh = utils.read_obj(smpl_path)
    uv_obj_path = os.path.join(indir, '%s_refine_uv.obj' % os.path.basename(indir))

    unwrapper = Unwrapper()
    uv_mesh = unwrapper.forward(reduced_body_mesh, smpl_mesh)
    utils.write_obj(uv_obj_path, uv_mesh)

    assets_dir = '../../models'
    texture = Texture(img_size=args.tex_size, assets_dir=assets_dir)
    kwargs = {'data_dir': data_dir, 'mesh_uv': uv_obj_path, 'use_idx': args.use_idx}
    tex_mesh = texture.forward(kwargs)
    uv_obj_tex_path = os.path.join(indir, '%s_refine_tex.obj' % os.path.basename(indir))
    utils.write_obj(uv_obj_tex_path, tex_mesh)

    tex = tex_mesh['texture_map']
    tex_path = os.path.join(indir, '%s_uv_tex.png' % os.path.basename(indir))
    cv2.imwrite(tex_path, tex)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--smpl_obj_path', default='refine/data/smpl.obj', type=str)
    parser.add_argument('--refine_path', default=None, type=str)
    parser.add_argument('--tex_size', default=512, type=int)
    parser.add_argument('--use_idx', default='0', type=str)


    args = parser.parse_args()


    main(args)
