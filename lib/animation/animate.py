# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import os
import cv2
from generate_skeleton import gen_skeleton_bvh
from utils import read_obj, write_obj
import os
import numpy as np

class Animator(object):

    def __init__(self, model_dir='assets', blender_path='blender'):
        self.model_dir = model_dir
        self.blender = blender_path
        print('model_dir:', self.model_dir)
        print('blender:', self.blender)

    def gen_skeleton(self, case_dir, action_dir, action):
        self.case_dir = case_dir
        self.action_dir = action_dir
        self.action = action
        status = gen_skeleton_bvh(self.model_dir, self.action_dir,
                                  self.case_dir, self.action)
        return status

    def gen_weights(self, save_dir=None):
        case_name = os.path.basename(self.case_dir)
        action_name = os.path.basename(self.action).replace('.npy', '')
        if save_dir is None:
            gltf_path = os.path.join(self.case_dir, '%s-%s.glb' %
                                     (case_name, action_name))
        else:
            os.makedirs(save_dir, exist_ok=True)
            gltf_path = os.path.join(save_dir, '%s-%s.glb' %
                                     (case_name, action_name))

        # current file directory name
        exec_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'skinning.py')

        cmd = f'{self.blender} -b -P {exec_path}  -- --input {self.case_dir}' \
              f' --gltf_path {gltf_path} --action {self.action}'
        print(cmd)
        os.system(cmd)
        return gltf_path

    def animate(self, mesh_path, action_dir, action, save_dir=None):
        case_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(mesh_path))), 'animation')
        os.makedirs(case_dir, exist_ok=True)
        tex_path = mesh_path.replace('.obj', '.png')
        mesh = read_obj(mesh_path)
        tex = cv2.imread(tex_path)
        vertices = mesh['vertices']
        trans = np.load(os.path.join(self.model_dir, '3D-assets/smpl_trans.npy'))
        vertices += trans
        mesh['vertices'] = vertices
        mesh['texture_map'] = tex
        outpath = os.path.join(case_dir, 'body.obj')
        write_obj(outpath, mesh)
        print('saved body mesh to %s' % outpath)

        self.gen_skeleton(case_dir, action_dir, action)
        gltf_path = self.gen_weights(save_dir)
        print('saved animation file to %s' % gltf_path)
        return gltf_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_path', default='body.obj', type=str)
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--action_dir', type=str, default='actions')
    parser.add_argument('--action', type=str, default='hiphop')

    args = parser.parse_args()

    model_dir = args.model_dir
    # install blender
    blender_file = os.path.join(model_dir, '3D-assets', 'blender-3.1.2-linux-x64.tar.xz')
    blender_path = os.path.join(model_dir, '3D-assets','blender-3.1.2-linux-x64', 'blender')
    if not os.path.exists(blender_file):
        raise Exception('found blender file failed.')
    if not os.path.exists(blender_path):
        cmd = f'tar -xvf {blender_file} -C {os.path.join(model_dir, "3D-assets")}'
        os.system(cmd)

    anim = Animator(model_dir, blender_path)

    mesh_path = args.mesh_path
    action_dir = args.action_dir
    action = args.action

    anim.animate(mesh_path, action_dir, action)
