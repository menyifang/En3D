# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import numpy as np
import trimesh
from matplotlib import cm as mpl_cm, colors as mpl_colors
import torch
from scipy.spatial import cKDTree
import argparse
import os
import shutil

def read_obj(obj_path, print_shape=False):
    try:
        with open(obj_path, 'r') as f:
            bfm_lines = f.readlines()
    except Exception as e:
        print("open mesh obj error: {}", e)
        return -1

    vertices = []
    faces = []
    uvs = []
    vns = []
    faces_uv = []
    faces_normal = []
    for line in bfm_lines:
        if line[:2] == 'v ':
            vertex = [float(a) for a in line.strip().split(' ')[1:] if len(a)>0]
            vertices.append(vertex)

        if line[:2] == 'f ':
            items = line.strip().split(' ')[1:]
            face = [int(a.split('/')[0]) for a in items if len(a)>0]
            if len(faces) > 0 and len(face) != len(faces[0]):
                continue
            faces.append(face)

            if '/' in items[0] and len(items[0].split('/'))>1 and len(items[0].split('/')[1])>0:
                face_uv = [int(a.split('/')[1]) for a in items if len(a)>0]
                faces_uv.append(face_uv)

            if '/' in items[0] and len(items[0].split('/'))>2 and len(items[0].split('/')[2])>0:
                face_normal = [int(a.split('/')[2]) for a in items if len(a)>0]
                faces_normal.append(face_normal)

        if line[:3] == 'vt ':
            items = line.strip().split(' ')[1:]
            uv = [float(a) for a in items if len(a)>0]
            uvs.append(uv)

        if line[:3] == 'vn ':
            items = line.strip().split(' ')[1:]
            vn = [float(a) for a in items if len(a)>0]
            vns.append(vn)

    vertices = np.array(vertices).astype(np.float32)
    faces = np.array(faces).astype(np.int32)

    if vertices.shape[1] == 3:
        mesh = {
            'vertices': vertices,
            'faces': faces,
        }
    else:
        mesh = {
            'vertices': vertices[:, :3],
            'faces': faces,
        }

    if len(uvs) > 0:
        uvs = np.array(uvs).astype(np.float32)
        mesh['uvs'] = uvs

    if len(vns) > 0:
        vns = np.array(vns).astype(np.float32)
        mesh['vns'] = vns

    if len(faces_uv) > 0:
        faces_uv = np.array(faces_uv).astype(np.int32)
        mesh['faces_uv'] = faces_uv

    if len(faces_normal) > 0:
        faces_normal = np.array(faces_normal).astype(np.int32)
        mesh['faces_normal'] = faces_normal

    if print_shape:
        print('vertices.shape', vertices.shape)
        print('faces.shape', faces.shape)
    return mesh


def part_segm_to_vertex_colors(part_segm, n_vertices, alpha=1.0):
    vertex_labels = np.zeros(n_vertices)

    for part_idx, (k, v) in enumerate(part_segm.items()):
        if k in ['leftHand', 'leftHandIndex1', 'rightHand', 'rightHandIndex1']:
            vertex_labels[v] = -1
        else:
            vertex_labels[v] = part_idx

    cm = mpl_cm.get_cmap('jet')
    norm_gt = mpl_colors.Normalize()

    vertex_colors = np.ones((n_vertices, 4))
    vertex_colors[:, 3] = alpha
    vertex_colors[:, :3] = cm(norm_gt(vertex_labels))[:, :3]

    return vertex_colors, vertex_labels

def repair_faces(vertices, faces):
    max = faces.max()
    len = vertices.shape[0]
    if max == len:
        faces -= 1
    return faces


def apply_face_mask(mesh, face_mask):
    mesh.update_faces(face_mask)
    mesh.remove_unreferenced_vertices()
    return mesh

def apply_vertex_mask(mesh, vertex_mask):
    faces_mask = vertex_mask[mesh.faces].any(dim=1)
    mesh = apply_face_mask(mesh, faces_mask)
    return mesh


def part_removal(full_mesh, part_mesh, thres, device, smpl_verts, smpl_vertex_labels, clean=True):

    smpl_tree = cKDTree(smpl_verts)

    from PointFeat import PointFeat

    part_extractor = PointFeat(
        torch.tensor(part_mesh.vertices).unsqueeze(0).to(device),
        torch.tensor(part_mesh.faces).unsqueeze(0).to(device))

    (part_dist, _) = part_extractor.query(torch.tensor(full_mesh.vertices).unsqueeze(0).to(device))

    remove_mask = part_dist < thres

    _, idx = smpl_tree.query(full_mesh.vertices, k=1)
    full_lmkid = smpl_vertex_labels[idx]
    remove_mask = torch.logical_and(remove_mask, torch.tensor(full_lmkid ==-1).type_as(remove_mask).unsqueeze(0))

    BNI_part_mask = ~(remove_mask).flatten()[full_mesh.faces].any(dim=1)
    full_mesh.update_faces(BNI_part_mask.detach().cpu())
    full_mesh.remove_unreferenced_vertices()

    return full_mesh

def get_submesh(verts, faces, verts_retained=None, faces_retained=None, min_vert_in_face=2):
    verts = verts
    faces = faces
    if verts_retained is not None:
        # Transform indices into bool array
        if verts_retained.dtype != 'bool':
            vert_mask = np.zeros(len(verts), dtype=bool)
            vert_mask[verts_retained] = True
        else:
            vert_mask = verts_retained
        # Faces with at least min_vert_in_face vertices
        bool_faces = np.sum(vert_mask[faces.ravel()].reshape(-1, 3), axis=1) > min_vert_in_face
    elif faces_retained is not None:
        # Transform indices into bool array
        if faces_retained.dtype != 'bool':
            bool_faces = np.zeros(len(faces_retained), dtype=bool)
        else:
            bool_faces = faces_retained
    new_faces = faces[bool_faces]
    # just in case additional vertices are added
    vertex_ids = list(set(new_faces.ravel()))
    oldtonew = -1 * np.ones([len(verts)])
    oldtonew[vertex_ids] = range(0, len(vertex_ids))
    new_verts = verts[vertex_ids]
    new_faces = oldtonew[new_faces].astype('int32')
    return (new_verts, new_faces)

def poisson_remesh(obj_path, save_path):
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_path)
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=200000) ### use 50w faces
    # ms.apply_coord_laplacian_smoothing()
    ms.save_current_mesh(save_path)
    ms.save_current_mesh(save_path.replace(".obj", ".ply"))
    polished_mesh = trimesh.load_mesh(save_path)

    return polished_mesh

def poisson(mesh, obj_path, noReduced_obj_path, ref_obj, depth=10):

    from pypoisson import poisson_reconstruction
    faces, vertices = poisson_reconstruction(mesh.vertices, mesh.vertex_normals, depth=depth)

    new_meshes = trimesh.Trimesh(vertices, faces)
    new_mesh_lst = new_meshes.split(only_watertight=False)
    comp_num = [new_mesh.vertices.shape[0] for new_mesh in new_mesh_lst]
    noReduced_mesh = new_mesh_lst[comp_num.index(max(comp_num))]
    noReduced_mesh.export(noReduced_obj_path)

    final_mesh = poisson_remesh(noReduced_obj_path, obj_path)

    return final_mesh

def expand_vert(vert, left_hand_xyz, left_xyz):
    left_qua = (left_hand_xyz+left_xyz)/2
    # left, down vertex of the vert
    x, y = (min(vert[:,0]), min(vert[:,1]))
    len_real = np.sqrt((left_qua[0]-x)**2+(left_qua[1]-y)**2)
    len_std = np.sqrt((left_qua[0]-left_hand_xyz[0])**2+(left_qua[1]-left_hand_xyz[1])**2)
    scale = len_std/len_real*1.2
    # print(scale)
    # print('left_qua', left_qua)
    # print(min(vert[:, 0]))
    # print(max(vert[:, 0]))

    move_idx = [i for i in range(vert.shape[0]) if vert[i, 0] < left_qua[0]]
    vert[move_idx, 0] = (vert[move_idx, 0]-left_qua[0])*scale+left_qua[0]
    move_idx = [i for i in range(vert.shape[0]) if vert[i, 1] < left_qua[1]]
    vert[move_idx, 1] = (vert[move_idx, 1]-left_qua[1])*scale+left_qua[1]

    return vert


def expad_distance_from_line(verts, k, b, t_pos):
    A = k
    B = -1
    C = b
    distances = np.abs(A * verts[:, 0] + B * verts[:, 1] + C) / np.sqrt(A ** 2 + B ** 2)
    std_dis = np.abs(A * t_pos[0] + B * t_pos[1] + C) / np.sqrt(A ** 2 + B ** 2)
    scale = ((std_dis/np.max(distances))-1)*1.1
    perp_direction = np.array([-k, 1])
    perp_direction = perp_direction / np.linalg.norm(perp_direction)
    sign = -(np.sign(A * verts[:, 0] + B * verts[:, 1] + C))
    new_xy = verts[:, :2] + perp_direction * distances[:, np.newaxis] * sign[:, np.newaxis] * scale
    new_verts = np.hstack((new_xy, verts[:, 2][:, np.newaxis]))

    return new_verts



class Refiner(object):
    def __init__(self, model_dir, body_path, smpl_path, merge_path, noReduce_merge_path, operation='refine_arm'):
        self.model_dir = model_dir
        self.body_path = body_path
        self.smpl_path = smpl_path
        self.merge_path = merge_path
        self.noReduce_merge_path = noReduce_merge_path
        self.operation = operation
        self.left_idx = [5533, 5483, 5571, 5559, 5562, 5550, 5495, 5527, 5663, 5662, 5645, 5690]
        self.right_idx = [2229, 2019, 2111, 2099, 2089, 2059, 2060, 2107]
        self.left_hand_joint = [5565, 5566, 5569, 5571, 5667, 5668, 5669]
        self.right_hand_joint = [2104, 2107, 2111, 2206, 2208, 2230, 2241]
        self.left_arm_joint = [5090, 5120, 5126, 5128, 5194, 5361]
        self.right_arm_joint = [1566, 1620, 1654, 1657, 1658, 1664, 1694, 1725]

        self.left_hand_params = [0.11147531, 0.06676591, 0.28420825, 0.31626157, 0.07217873, 0.012502, 0.13660822]
        self.right_hand_params = [0.38022787, 0.14419409, 0.08651072, 0.07735174, 0.16368477, 0.13764798, 0.01038283]
        self.left_arm_params = [0.22084812, 0.19964834, 0.09163094, 0.07941337, 0.24966386, 0.15879537]
        self.right_arm_params = [0.06388703, 0.00131848, 0.13791009, 0.35664698, 0.22623685, 0.11176222, 0.05803539, 0.04420295]

        self.left_should = [695, 1463, 4123, 4124, 6403, 6460, 6468]
        self.left_should_params = [0.0252665, 0.00189856, 0.07650198, 0.0134348, 0.09136475, 0.36714087, 0.42439254]
        self.right_should = [637, 1255, 1256, 1873, 1891, 2878, 2944, 3000, 3008, 3009, 6534]
        self.right_should_params = [0.07228735, 0.01739097, 0.0033528, 0.06756481, 0.17250087, 0.02213909, 0.21043813,
                               0.05194716, 0.04983415, 0.32932983, 0.00321482]

    def tune_pos_hand(self, hand_mesh):
        data = np.load(os.path.join(os.path.dirname(self.body_path), 'arm.npy')) # 6,3
        delta = (abs(data[2,2]-data[0,2])+abs(data[5,2]-data[3,2]))/2
        hand_mesh.vertices[:,2] -= delta
        return hand_mesh


    def process(self):
        body_path = self.body_path
        hand_path = self.smpl_path
        use_poisson = True

        smpl_obj = read_obj(hand_path)
        smpl_obj['faces'] = repair_faces(smpl_obj['vertices'], smpl_obj['faces'])
        smpl_mesh = trimesh.Trimesh(smpl_obj['vertices'], smpl_obj['faces'], process=False, maintains_order=True)


        part_segm = json.load(open(os.path.join(self.model_dir, 'smpl_vert_segmentation.json')))
        # print(part_segm.keys())
        left_hand_idxs = []
        right_hand_idxs = []
        hand_idxs = []
        for key in ['leftHand', 'leftHandIndex1']:
            left_hand_idxs += part_segm[key]
        for key in ['rightHand', 'rightHandIndex1']:
            right_hand_idxs += part_segm[key]
        for key in ['leftHand', 'leftHandIndex1', 'rightHand', 'rightHandIndex1']:
            hand_idxs += part_segm[key]

        body_obj = read_obj(body_path)
        body_obj['faces'] = repair_faces(body_obj['vertices'], body_obj['faces'])

        # clean mesh
        mesh_lst = trimesh.Trimesh(body_obj['vertices'], body_obj['faces'])
        mesh_lst = mesh_lst.split(only_watertight=False)
        comp_num = [mesh.vertices.shape[0] for mesh in mesh_lst]
        mesh_clean = mesh_lst[comp_num.index(max(comp_num))]
        verts = mesh_clean.vertices
        faces = mesh_clean.faces  # start from 0
        body_obj['vertices'] = verts
        body_obj['faces'] = faces

        # print(hand_idxs)
        hand_mask = torch.zeros(smpl_obj['vertices'].shape[0], )
        hand_mask.index_fill_(0, torch.tensor(hand_idxs), 1.0)

        hand_mesh = apply_vertex_mask(smpl_mesh.copy(), hand_mask)

        # tune hand position
        tune_hand_pos = False
        if tune_hand_pos:
            vertices = hand_mesh.vertices
            right_hand_idxs = [i for i in range(vertices.shape[0]) if vertices[i, 0] < 0]
            vertices[right_hand_idxs, 1] -= 0.02
            hand_mesh.vertices = vertices


        if self.operation == "refine_arm":
            # print("refine arm")
            left_should_joint = smpl_obj['vertices'][self.left_should]
            left_should_xyz = np.sum(left_should_joint * np.array(self.left_should_params)[:, np.newaxis], axis=0)
            right_should_joint = smpl_obj['vertices'][self.right_should]
            right_should_xyz = np.sum(right_should_joint * np.array(self.right_should_params)[:, np.newaxis], axis=0)

            leftarm_joint = smpl_obj['vertices'][self.left_arm_joint]
            left_xyz = np.sum(leftarm_joint * np.array(self.left_arm_params)[:, np.newaxis], axis=0)
            rightarm_joint = smpl_obj['vertices'][self.right_arm_joint]
            right_xyz = np.sum(rightarm_joint * np.array(self.right_arm_params)[:, np.newaxis], axis=0)

            left_idxjoint = smpl_obj['vertices'][self.left_idx]
            right_idxjoint = smpl_obj['vertices'][self.right_idx]
            left_max = np.max(left_idxjoint, axis=0)
            right_max = np.max(right_idxjoint, axis=0)
            left_min = np.min(left_idxjoint, axis=0)
            right_min = np.min(right_idxjoint, axis=0)
            left_hand_size = (left_max - left_min)
            right_hand_size = (right_max - right_min)
            left_hand_xyz = (left_max + left_min) / 2
            right_hand_xyz = (right_max + right_min) / 2
            if tune_hand_pos:
                left_hand_xyz[1] -= 0.02
            step = 0.005

            vertices = body_obj['vertices']

            #compute k and d for xy
            kl = (left_xyz[1] - left_hand_xyz[1]) / (left_xyz[0] - left_hand_xyz[0])
            kr = (right_xyz[1] - right_hand_xyz[1]) / (right_xyz[0] - right_hand_xyz[0])
            dl = (1 / kl) * left_hand_xyz[0] + left_hand_xyz[1]
            dr = (1 / kr) * right_hand_xyz[0] + right_hand_xyz[1]
            kl2 = (left_should_xyz[1] - left_hand_xyz[1]) / (left_should_xyz[0] - left_xyz[0])
            kr2 = (right_should_xyz[1] - right_hand_xyz[1]) / (right_should_xyz[0] - right_xyz[0])

            left_hand_cur = left_hand_xyz +[0,0,-0.01]
            left_fore_half = (left_hand_xyz+left_xyz)/2
            left_seg_mm_half = (((-1 / kl) * (vertices[:, 0] - left_hand_xyz[0]) - (vertices[:, 1] - left_hand_xyz[1])) < 0) \
                          & (((-1 / kl) * (vertices[:, 0] - left_fore_half[0]) - (vertices[:, 1] - left_fore_half[1])) > 0) \
                          & (kl2 * (vertices[:, 0] - left_xyz[0]) - (vertices[:, 1] - left_hand_xyz[1]) < 0)
            left_vert_half = vertices[left_seg_mm_half]

            # expan left_vert_half along line y=kx+b
            kl_cur = (left_xyz[1] - left_hand_cur[1]) / (left_xyz[0] - left_hand_cur[0])
            left_vert_half = expad_distance_from_line(left_vert_half, -1/kl_cur, (1 / kl_cur) * left_fore_half[0] + left_fore_half[1],
                                                      left_hand_xyz)
            vertices[left_seg_mm_half] = left_vert_half

            left_seg_mm = (((-1 / kl) * (vertices[:, 0] - left_hand_xyz[0]) - (
                        vertices[:, 1] - left_hand_xyz[1])) < 0) & (((-1 / kl) * (vertices[:, 0] - left_xyz[0]) - (
                        vertices[:, 1] - left_xyz[1])) > 0) & (
                                  kl2 * (vertices[:, 0] - left_xyz[0]) - (vertices[:, 1] - left_hand_xyz[1]) < 0)
            right_seg_mm = (((-1 / kr) * (vertices[:, 0] - right_hand_xyz[0]) - (
                        vertices[:, 1] - right_hand_xyz[1])) < 0) & (((-1 / kr) * (vertices[:, 0] - right_xyz[0]) - (
                        vertices[:, 1] - right_xyz[1])) > 0) & (
                                   kr2 * (vertices[:, 0] - right_xyz[0]) - (vertices[:, 1] - right_hand_xyz[1]) < 0)
            left_vert = vertices[left_seg_mm]

            right_vert = vertices[right_seg_mm]
            len_left = pow(pow(left_xyz[0] - left_hand_xyz[0], 2) + pow(left_xyz[1] - left_hand_xyz[1], 2), 0.5)
            len_right = pow(pow(right_xyz[0] - right_hand_xyz[0], 2) + pow(right_xyz[1] - right_hand_xyz[1], 2), 0.5)
            tt_left = int(len_left // step)
            tt_right = int(len_right // step)

            center_ll = []
            center_rr = []
            len_ll = []
            len_rr = []
            dis = np.abs((-1 / kl) * left_vert[:, 0] - left_vert[:, 1] + dl) / pow(1 + pow(-1 / kl, 2), 0.5)
            for i in range(tt_left):
                ll_mm = (dis < (step * (i + 1))) & (dis > (step * i))
                ll_max = np.max(left_vert[ll_mm], axis=0)
                ll_min = np.min(left_vert[ll_mm], axis=0)
                center_ll1 = (ll_max + ll_min) / 2
                center_ll.append(center_ll1)
                len_ll.append(ll_max - ll_min)
            dis = np.abs((-1 / kr) * right_vert[:, 0] - right_vert[:, 1] + dr) / pow(1 + pow(-1 / kr, 2), 0.5)
            for i in range(tt_right):
                rr_mm = (dis < (step * (i + 1))) & (dis > (step * i))
                rr_max = np.max(right_vert[rr_mm], axis=0)
                rr_min = np.min(right_vert[rr_mm], axis=0)
                center_rr1 = (rr_max + rr_min) / 2
                center_rr.append(center_rr1)
                len_rr.append(rr_max - rr_min)

            left_fix = left_vert * 1.
            right_fix = right_vert * 1.
            disll = np.abs((-1 / kl) * left_vert[:, 0] - left_vert[:, 1] + dl) / pow(1 + pow(-1 / kl, 2), 0.5)
            left_fix = left_fix[disll < (step * tt_left)]
            disrr = np.abs((-1 / kr) * right_vert[:, 0] - right_vert[:, 1] + dr) / pow(1 + pow(-1 / kr, 2), 0.5)
            right_fix = right_fix[disrr < (step * tt_right)]

            #hand size fix
            z_len_ll = np.array(len_ll)
            z_center_ll = np.array(center_ll)
            disll_fix = np.abs((-1 / kl) * left_fix[:, 0] - left_fix[:, 1] + dl) / pow(1 + pow(-1 / kl, 2), 0.5)
            fix_z = z_len_ll[((disll_fix / step).astype(np.int8)).tolist()]
            fix_cen = z_center_ll[((disll_fix / step).astype(np.int8)).tolist()]
            left_fix[:, 2] = (disll_fix / (step * tt_left) +
                              (left_hand_size[2] / fix_z[:, 2]) * ((step * tt_left) - disll_fix) / (step * tt_left)) \
                             * (left_fix[:, 2] - fix_cen[:, 2]) \
                             + fix_cen[:, 2]


            z_len_rr = np.array(len_rr)
            z_center_rr = np.array(center_rr)
            disrr_fix = np.abs((-1 / kr) * right_fix[:, 0] - right_fix[:, 1] + dr) / pow(1 + pow(-1 / kr, 2), 0.5)
            fix_z = z_len_rr[((disrr_fix / step).astype(np.int8)).tolist()]
            fix_cen = z_center_rr[((disrr_fix / step).astype(np.int8)).tolist()]
            right_fix[:, 2] = (disrr_fix / (step * tt_right) + (
                    right_hand_size[2] / fix_z[:, 2]) * ((step * tt_right) - disrr_fix) / (step * tt_right)) * (
                                      right_fix[:, 2] - fix_cen[:, 2]) + fix_cen[:, 2]

            #hand pose fix
            z_left_hand = left_hand_xyz[2] - center_ll[0][2]
            z_right_hand = right_hand_xyz[2] - center_rr[0][2]
            # print(z_left_hand, z_right_hand)
            left_z = (step * tt_left - disll_fix) / (step * tt_left)
            right_z = (step * tt_right - disrr_fix) / (step * tt_right)
            left_fix[:, 2] = left_z.clip(0., 1.) * z_left_hand + left_fix[:, 2]
            right_fix[:, 2] = right_z.clip(0., 1.) * z_right_hand + right_fix[:, 2]
            z_left_hand = left_hand_xyz[1] - center_ll[0][1]
            z_right_hand = right_hand_xyz[1] - center_rr[0][1]
            left_fix[:, 1] = left_z.clip(0., 1.) * z_left_hand + left_fix[:, 1]
            right_fix[:, 1] = right_z.clip(0., 1.) * z_right_hand + right_fix[:, 1]

            left_vert[disll < (step * tt_left)] = left_fix
            right_vert[disrr < (step * tt_right)] = right_fix
            vertices[left_seg_mm] = left_vert
            vertices[right_seg_mm] = right_vert
            body_obj['vertices'] = vertices

        full_lst = []
        full_lst += [hand_mesh]

        _, smpl_vertex_labels = part_segm_to_vertex_colors(part_segm, smpl_obj['vertices'].shape[0])

        body_mesh = trimesh.Trimesh(body_obj['vertices'], body_obj['faces'], process=False, maintains_order=True)

        # remove hand neighbor triangles
        if self.operation == "refine_arm":
            left_seg_mm = (((-1 / kl) * (body_obj['vertices'][:, 0] - left_hand_xyz[0]) - (
                        body_obj['vertices'][:, 1] - left_hand_xyz[1])) < 0) | (
                                  kl2 * (body_obj['vertices'][:, 0] - left_hand_xyz[0]) - (body_obj['vertices'][:, 1]) > 0)

            right_seg_mm = (((-1 / kr) * (body_obj['vertices'][:, 0] - right_hand_xyz[0]) - (
                        body_obj['vertices'][:, 1] - right_hand_xyz[1])) < 0) | (
                                   kr2 * (body_obj['vertices'][:, 0] - right_hand_xyz[0]) - (body_obj['vertices'][:, 1]) > 0)
            seg_mm = left_seg_mm & right_seg_mm
            new_verts, new_faces = get_submesh(body_obj['vertices'], body_obj['faces'], verts_retained=seg_mm)
            body_mesh = trimesh.Trimesh(new_verts, new_faces, process=False, maintains_order=True)
            device = torch.device(f"cuda:0")
            body_mesh = part_removal(body_mesh, hand_mesh, 8e-2, device, smpl_obj['vertices'], smpl_vertex_labels)
            full_lst += [body_mesh]
        else:
            device = torch.device(f"cuda:0")
            body_mesh = part_removal(body_mesh, hand_mesh, 8e-2, device, smpl_obj['vertices'], smpl_vertex_labels)
            full_lst += [body_mesh]

        final_path = self.merge_path

        if use_poisson:
            final_mesh = poisson(
                sum(full_lst),
                final_path,
                self.noReduce_merge_path,
                body_obj
            )
            print('save to %s' % final_path)
        else:
            final_mesh = sum(full_lst)
            final_mesh.export(final_path)
            final_mesh.export(self.noReduce_merge_path)
            print('save to %s' % final_path)


def main(args):
    model_dir = args.model_dir
    body_path = args.mesh_path
    smpl_path = args.smpl_obj_path
    operation = args.operation
    indir = os.path.dirname(body_path)
    if args.out_dir is None:
        refine_path = os.path.join(indir, '%s_refine.obj' % os.path.basename(indir))
        noReduce_refine_path = os.path.join(indir, '%s_refine_noReduce.obj' % os.path.basename(indir))
    else:
        case_name = args.mesh_path.split('/')[-5]
        refine_path = os.path.join(args.out_dir, '%s_refine.obj' % case_name)
        noReduce_refine_path = os.path.join(args.out_dir, '%s_refine_noReduce.obj' % case_name)

    print('start refine hand and remesh ...')

    algo = Refiner(model_dir, body_path, smpl_path, refine_path, noReduce_refine_path, operation)
    algo.process()

    print('refine finished!')
    return refine_path, noReduce_refine_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', default='../../models', type=str)
    parser.add_argument('--mesh_path', default='assets', type=str)
    parser.add_argument('--smpl_obj_path', default='assets', type=str)
    parser.add_argument('--operation', default='refine_arm', type=str)
    parser.add_argument('--out_dir', default=None, type=str)

    args = parser.parse_args()

    main(args)


