# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import utils
import numpy as np
import math
import torch
import cv2
import time
import argparse
import trimesh

def split_body(body_mesh, smpl_mesh):
    body_vertices = body_mesh['vertices']
    smpl_vertices = smpl_mesh['vertices']

    n_body_vertices = len(body_vertices)

    smpl_neck_bottom_ind = 3168
    smpl_mid_bottom_ind = 1210
    smpl_underbelly_ind = 1807

    x = body_vertices[:, 0]
    y = body_vertices[:, 1]
    x1 = smpl_vertices[smpl_neck_bottom_ind, 0]
    y1 = smpl_vertices[smpl_neck_bottom_ind, 1]

    arm_bias = 0.2

    # right arm
    smpl_r_armpit_ind = 4171
    smpl_r_mid_fingertip_ind = 5905
    smpl_r_elbow_front_ind = 5118
    smpl_r_elbow_back_ind = 5214
    x2 = smpl_vertices[smpl_r_mid_fingertip_ind, 0]
    body_r_arm_inds = \
        np.where((x < smpl_vertices[smpl_r_armpit_ind, 0]-0.01) *
                 (y - y1 * (x - x2) / (x1 - x2) + arm_bias > 0))[0]


    # left arm
    smpl_l_armpit_ind = 1761
    smpl_l_mid_fingertip_ind = 2445
    x = body_vertices[:, 0]
    y = body_vertices[:, 1]
    smpl_l_elbow_front_ind = 1749
    smpl_l_elbow_back_ind = 1705
    x2 = smpl_vertices[smpl_l_mid_fingertip_ind, 0]
    body_l_arm_inds = \
        np.where((x > smpl_vertices[smpl_l_armpit_ind, 0]) *
                 (y - y1 * (x - x2) / (x1 - x2) + arm_bias > 0))[0]


    leg_bias = 0.07

    # right leg
    smpl_r_knee_front_ind = 4529
    smpl_r_knee_back_ind = 4486
    x2 = smpl_vertices[smpl_r_mid_fingertip_ind, 0]
    body_r_lower_leg_inds = np.where((y < smpl_vertices[smpl_r_knee_front_ind, 1]+leg_bias) *
                                     (x < smpl_vertices[smpl_mid_bottom_ind, 0]) * (
                                             y - y1 * (x - x2) / (x1 - x2) + arm_bias < 0))[0]

    # left leg
    smpl_l_knee_front_ind = 1019
    smpl_l_knee_back_ind = 1002
    x2 = smpl_vertices[smpl_l_mid_fingertip_ind, 0]
    body_l_lower_leg_inds = np.where((y < smpl_vertices[smpl_l_knee_front_ind, 1]+leg_bias) *
                                     (x >= smpl_vertices[smpl_mid_bottom_ind, 0]) * (
                                             y - y1 * (x - x2) / (x1 - x2) + arm_bias < 0))[0]

    # main body
    body_main_inds = set(range(n_body_vertices)) - set(body_r_arm_inds) - set(body_l_arm_inds) - \
                     set(body_r_lower_leg_inds) - set(body_l_lower_leg_inds)
    body_main_inds = list(body_main_inds)

    gap = 0.01

    body_parts = {
        'main': {
            'inds': np.array(body_main_inds, dtype=np.int64),
            'top': (smpl_vertices[3935] + smpl_vertices[441]) / 2,  # 脖子中间
            'bottom': (smpl_vertices[4165] + smpl_vertices[677]) / 2,  # 肚子中间
            # 'bottom': (smpl_vertices[smpl_r_knee_front_ind] + smpl_vertices[smpl_r_knee_back_ind]) / 2,  # 右膝盖中间
            'uv_range': [1 / 6 + gap, 5 / 6 - gap, 0, 1.]  # u_start, u_end, v_start, v_end
        },
        'right_arm': {
            'inds': np.array(body_r_arm_inds, dtype=np.int64),
            'top': smpl_vertices[4271],  # 右肩膀
            'bottom': (smpl_vertices[5563] + smpl_vertices[5669]) / 2,  # 右手腕中间
            'uv_range': [0., 1 / 6 - gap, 1 / 2 + gap, 1.]
        },
        'leftr_arm': {
            'inds': np.array(body_l_arm_inds, dtype=np.int64),
            'top': smpl_vertices[782],  # 左肩膀
            'bottom': (smpl_vertices[2103] + smpl_vertices[2208]) / 2,  # 左手腕中间
            'uv_range': [5 / 6 + gap, 1., 1 / 2 + gap, 1.]
        },
        'right_lower_leg': {
            'inds': np.array(body_r_lower_leg_inds, dtype=np.int64),
            'top': (smpl_vertices[smpl_r_knee_front_ind] + smpl_vertices[smpl_r_knee_back_ind]) / 2 ,  # 右膝盖中间
            'bottom': smpl_vertices[6858],  # 右脚根
            'uv_range': [0., 1 / 6 - gap, 0., 1 / 2 - gap]
        },
        'left_lower_leg': {
            'inds': np.array(body_l_lower_leg_inds, dtype=np.int64),
            'top': (smpl_vertices[smpl_l_knee_front_ind] + smpl_vertices[smpl_l_knee_back_ind]) / 2 ,  # 左膝盖中间
            'bottom': smpl_vertices[3458],  # 左脚根
            'uv_range': [5 / 6 + gap, 1., 0., 1 / 2 - gap]
        },
    }

    return body_parts

def align_to_y_axis(vertices, top_vert, bottom_vert):
    vertices_direction = top_vert - bottom_vert
    x1, y1, z1 = vertices_direction[0], vertices_direction[1], vertices_direction[2]
    x_degree = - np.arctan(z1 / y1)
    z_degree = np.pi / 2 - np.arccos(x1 / np.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2))

    lmk_verts = np.stack([top_vert, bottom_vert], axis=0)
    vertices = np.concatenate([vertices.copy(), lmk_verts], axis=0)

    vertices = utils.rotate_vertices(vertices, [x_degree, 0, 0])
    vertices = utils.rotate_vertices(vertices, [0, 0, z_degree])

    top_vert = vertices[-2]
    bottom_vert = vertices[-1]
    vertices = vertices[:-2]

    return vertices, top_vert, bottom_vert


def cylinder_unwrapping(vertices, top_vert, bottom_vert, name=None):

    os.makedirs('tmp', exist_ok=True)
    mesh = {'vertices': np.concatenate([vertices, np.stack([top_vert, bottom_vert], axis=0)], axis=0)}
    utils.write_obj(os.path.join('tmp', '{}_split.obj'.format(name)), mesh)


    # recenter
    if name == 'main' or name == 'head':
        vertices[:, 2] -= 0.01
    else:

        vertices, top_vert, bottom_vert = align_to_y_axis(vertices, top_vert, bottom_vert)

        mesh = {'vertices': np.concatenate([vertices, np.stack([top_vert, bottom_vert], axis=0)], axis=0)}
        utils.write_obj(os.path.join('tmp', '{}_aligned.obj'.format(name)), mesh)


        vertices[:, 0] -= top_vert[0]
        vertices[:, 2] -= top_vert[2]


    mesh = {'vertices': np.concatenate([vertices, np.stack([top_vert, bottom_vert], axis=0)], axis=0)}
    utils.write_obj(os.path.join('tmp', '{}_aligned_recenter.obj'.format(name)), mesh)


    max_y = np.max(vertices[:, 1])
    min_y = np.min(vertices[:, 1])
    radius = vertices[:, 0] ** 2 + vertices[:, 2] ** 2
    radius_squared = np.max(radius)
    radius = math.sqrt(radius_squared)

    cylinder_R = np.array(radius * 1.1, dtype=np.float32)

    cylinder_verts = cylinder_R * vertices / np.sqrt(vertices[:, :1] ** 2 + vertices[:, 2:] ** 2)
    cylinder_verts = cylinder_verts.clip(-cylinder_R, cylinder_R)
    cylinder_verts[:, 1] = vertices[:, 1]

    xyz_2_uv = []
    idx = 0
    for xyzrgb in cylinder_verts:
        if xyzrgb[0] <= 0:
            theta = math.acos(-xyzrgb[2] / cylinder_R)
        else:
            theta = 2 * math.pi - math.acos(- xyzrgb[2] / cylinder_R)

        norm_u = theta / (2 * math.pi)
        norm_v = abs(xyzrgb[1] - max_y) / (max_y - min_y)
        xyz_2_uv.append(vertices[idx].tolist() + [norm_u, 1.0 - norm_v])
        idx += 1

    uvs = np.array(xyz_2_uv)[:, -2:]

    return uvs


def refine_faces_uv(UVs, faces_uv):
    line_to_vert_mapping = {}
    for face in faces_uv:
        key = '{}_{}'.format(min(face[0], face[1]) - 1, max(face[0], face[1]) - 1)
        if key not in line_to_vert_mapping:
            line_to_vert_mapping[key] = []
        line_to_vert_mapping[key].append(face[2] - 1)

        key = '{}_{}'.format(min(face[0], face[2]) - 1, max(face[0], face[2]) - 1)
        if key not in line_to_vert_mapping:
            line_to_vert_mapping[key] = []
        line_to_vert_mapping[key].append(face[1] - 1)

        key = '{}_{}'.format(min(face[1], face[2]) - 1, max(face[1], face[2]) - 1)
        if key not in line_to_vert_mapping:
            line_to_vert_mapping[key] = []
        line_to_vert_mapping[key].append(face[0] - 1)

    new_faces_uv = []
    thresh = 0.05
    for face in faces_uv:
        uv0 = UVs[face[0] - 1]
        uv1 = UVs[face[1] - 1]
        uv2 = UVs[face[2] - 1]

        if np.max(np.abs(uv0 - uv1)) > thresh and np.max(np.abs(uv0 - uv2)) > thresh:
            candidates = line_to_vert_mapping['{}_{}'.format(min(face[1], face[2]) - 1, max(face[1], face[2]) - 1)]
            new_v = candidates[0] if face[0] - 1 == candidates[1] else candidates[1]
            new_v += 1
            new_face = [new_v, face[1], face[2]]
        elif np.max(np.abs(uv1 - uv0)) > thresh and np.max(np.abs(uv1 - uv2)) > thresh:
            candidates = line_to_vert_mapping['{}_{}'.format(min(face[0], face[2]) - 1, max(face[0], face[2]) - 1)]
            new_v = candidates[0] if face[1] - 1 == candidates[1] else candidates[1]
            new_v += 1
            new_face = [face[0], new_v, face[2]]
        elif np.max(np.abs(uv2 - uv0)) > thresh and np.max(np.abs(uv2 - uv1)) > thresh:
            candidates = line_to_vert_mapping['{}_{}'.format(min(face[0], face[1]) - 1, max(face[0], face[1]) - 1)]
            new_v = candidates[0] if face[2] - 1 == candidates[1] else candidates[1]
            new_v += 1
            new_face = [face[0], face[1], new_v]
        else:
            new_face = face

        uv0 = UVs[new_face[0] - 1]
        uv1 = UVs[new_face[1] - 1]
        uv2 = UVs[new_face[2] - 1]
        if np.max(np.abs(uv0 - uv1)) > thresh or np.max(np.abs(uv0 - uv2)) > thresh or np.max(
                np.abs(uv1 - uv2)) > thresh:
            new_face = [face[0], face[0], face[0]]

        new_faces_uv.append(new_face)

    new_faces_uv = np.array(new_faces_uv)
    return new_faces_uv


def compute_body_uv(body_mesh, smpl_mesh):
    body_parts = split_body(body_mesh, smpl_mesh)
    ind_array = []
    uv_array = []
    for part_name in body_parts:
        print(part_name)
        UVs = cylinder_unwrapping(body_mesh['vertices'].copy()[body_parts[part_name]['inds']],
                                       body_parts[part_name]['top'],
                                       body_parts[part_name]['bottom'], part_name)
        uv_range = body_parts[part_name]['uv_range']
        UVs[:, 0] = uv_range[0] + UVs[:, 0] * (uv_range[1] - uv_range[0])
        UVs[:, 1] = uv_range[2] + UVs[:, 1] * (uv_range[3] - uv_range[2])
        body_parts[part_name]['UVs'] = UVs
        ind_array.append(body_parts[part_name]['inds'])
        uv_array.append(body_parts[part_name]['UVs'])
    ind_array = np.concatenate(ind_array, axis=0)
    uv_array = np.concatenate(uv_array, axis=0)
    uv_array = uv_array[np.argsort(ind_array)]

    t1 = time.time()
    faces_uv = refine_faces_uv(uv_array, body_mesh['faces'])
    print('refine_faces_uv', time.time() - t1)

    return uv_array, faces_uv




class Unwrapper(object):

    def __init__(self):
        print('Unwrapper init done.')

    def forward(self, reduced_body_mesh, smpl_mesh):
        print('total vertices: ', reduced_body_mesh['vertices'].shape)
        # textured_ori_mesh = colored_to_textured_mesh(ori_body_mesh, smpl_mesh)
        reduced_mesh_UVs, reduced_mesh_faces_uv = compute_body_uv(reduced_body_mesh, smpl_mesh)

        mesh_uv = {
            'vertices': reduced_body_mesh['vertices'],
            'faces': reduced_body_mesh['faces'],
            'uvs': reduced_mesh_UVs,
            'faces_uv': reduced_mesh_faces_uv,
            # 'texture_map': textured_ori_mesh['texture_map'],
        }

        return mesh_uv


def main(args):
    smpl_path = args.smpl_obj_path
    refine_path = args.refine_path
    indir = os.path.dirname(refine_path)

    assert smpl_path[-4:] == '.obj'
    assert refine_path[-4:] == '.obj'

    reduced_body_mesh = utils.read_obj(refine_path)
    smpl_mesh = utils.read_obj(smpl_path)

    unwrapper = Unwrapper()
    textured_mesh = unwrapper.forward(reduced_body_mesh, smpl_mesh)

    textured_obj_path = os.path.join(indir, '%s_refine_uv.obj' % os.path.basename(indir))
    texture_path = textured_obj_path.replace('.obj', '.jpg')
    utils.write_obj(textured_obj_path, textured_mesh)

    return textured_obj_path, texture_path

