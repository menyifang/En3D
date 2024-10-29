# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
import os
import cv2
import torch
import torch.nn.functional as F


def read_obj(obj_path, print_shape=False):
    with open(obj_path, 'r') as f:
        bfm_lines = f.readlines()

    vertices = []
    faces = []
    uvs = []
    vns = []
    faces_uv = []
    faces_normal = []
    max_face_length = 0
    for line in bfm_lines:
        if line[:2] == 'v ':
            vertex = [float(a) for a in line.strip().split(' ')[1:] if len(a)>0]
            vertices.append(vertex)

        if line[:2] == 'f ':
            items = line.strip().split(' ')[1:]
            face = [int(a.split('/')[0]) for a in items if len(a)>0]
            max_face_length = max(max_face_length, len(face))
            # if len(faces) > 0 and len(face) != len(faces[0]):
            #     continue
            faces.append(face)

            if '/' in items[0] and len(items[0].split('/')[1])>0:
                face_uv = [int(a.split('/')[1]) for a in items if len(a)>0]
                faces_uv.append(face_uv)

            if '/' in items[0] and len(items[0].split('/')) >= 3 and len(items[0].split('/')[2])>0:
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
    if max_face_length <= 3:
        faces = np.array(faces).astype(np.int32)
    else:
        print('not a triangle face mesh!')

    if vertices.shape[1] == 3:
        mesh = {
            'vertices': vertices,
            'faces': faces,
        }
    else:
        mesh = {
            'vertices': vertices[:, :3],
            'colors': vertices[:, 3:],
            'faces': faces,
        }

    if len(uvs) > 0:
        uvs = np.array(uvs).astype(np.float32)
        mesh['uvs'] = uvs

    if len(vns) > 0:
        vns = np.array(vns).astype(np.float32)
        mesh['normals'] = vns

    if len(faces_uv) > 0:
        if max_face_length <= 3:
            faces_uv = np.array(faces_uv).astype(np.int32)
        mesh['faces_uv'] = faces_uv

    if len(faces_normal) > 0:
        if max_face_length <= 3:
            faces_normal = np.array(faces_normal).astype(np.int32)
        mesh['faces_normal'] = faces_normal

    if print_shape:
        print('num of vertices', len(vertices))
        print('num of faces', len(faces))
    return mesh


def write_obj(save_path, mesh):
    save_dir = os.path.dirname(save_path)
    save_name = os.path.splitext(os.path.basename(save_path))[0]

    if 'texture_map' in mesh:
        cv2.imwrite(
            os.path.join(save_dir, save_name + '.png'), mesh['texture_map'])

        with open(os.path.join(save_dir, save_name + '.mtl'), 'w') as wf:
            wf.write('newmtl material_0\n')
            wf.write('Ka 1.000000 0.000000 0.000000\n')
            wf.write('Kd 1.000000 1.000000 1.000000\n')
            wf.write('Ks 0.000000 0.000000 0.000000\n')
            wf.write('Tr 0.000000\n')
            wf.write('illum 0\n')
            wf.write('Ns 0.000000\n')
            wf.write('map_Kd {}\n'.format(save_name + '.png'))

    with open(save_path, 'w') as wf:
        if 'texture_map' in mesh:
            wf.write('# Create by ModelScope\n')
            wf.write('mtllib ./{}.mtl\n'.format(save_name))

        if 'colors' in mesh:
            for i, v in enumerate(mesh['vertices']):
                wf.write('v {} {} {} {} {} {}\n'.format(
                    v[0], v[1], v[2], mesh['colors'][i][0],
                    mesh['colors'][i][1], mesh['colors'][i][2]))
        else:
            for v in mesh['vertices']:
                wf.write('v {} {} {}\n'.format(v[0], v[1], v[2]))

        if 'uvs' in mesh:
            for uv in mesh['uvs']:
                wf.write('vt {} {}\n'.format(uv[0], uv[1]))

        if 'normals' in mesh:
            for vn in mesh['normals']:
                wf.write('vn {} {} {}\n'.format(vn[0], vn[1], vn[2]))

        if 'faces' in mesh:
            for ind, face in enumerate(mesh['faces']):
                if 'faces_uv' in mesh or 'faces_normal' in mesh:
                    if 'faces_uv' in mesh:
                        face_uv = mesh['faces_uv'][ind]
                    else:
                        face_uv = face
                    if 'faces_normal' in mesh:
                        face_normal = mesh['faces_normal'][ind]
                    else:
                        face_normal = face
                    row = 'f ' + ' '.join([
                        '{}/{}/{}'.format(face[i], face_uv[i], face_normal[i])
                        for i in range(len(face))
                    ]) + '\n'
                else:
                    row = 'f ' + ' '.join(
                        ['{}'.format(face[i])
                         for i in range(len(face))]) + '\n'
                wf.write(row)


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    # rel_joints[:, 1:] -= joints[:, parents[1:]]

    # transforms_mat = transform_mat(
    #     rot_mats.view(-1, 3, 3),
    #     rel_joints.view(-1, 3, 1)).view(-1, joints.shape[1], 4, 4)
    transforms_mat = transform_mat(
        rot_mats.view(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    # for i in range(1, parents.shape[0]):
    #     # Subtract the joint location at the rest pose
    #     # No need for rotation, since it's identity when at rest
    #     curr_res = torch.matmul(transform_chain[parents[i]],
    #                             transforms_mat[:, i])
    #     transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms


def rotate_vertices(verts, rotate_degree):
    vertices = torch.from_numpy(verts).float()[None, ...]  # (1, n, 3)

    center = np.array([0, 0, 0])[None, None, ...]
    center = torch.from_numpy(center).float()

    rot_coeffs = np.array(rotate_degree)[None, None, ...]
    rot_coeffs = torch.from_numpy(rot_coeffs).float()

    rot_mats = batch_rodrigues(rot_coeffs.view(-1, 3), dtype=torch.float32).view([1, -1, 3, 3])

    J_transformed, A = batch_rigid_transform(rot_mats, center)

    homogen_coord = torch.ones([1, vertices.shape[1], 1], dtype=torch.float32)
    v_posed_homo = torch.cat([vertices, homogen_coord], dim=2)
    v_homo = torch.matmul(A, torch.unsqueeze(v_posed_homo, dim=-1))

    rot_vertices = v_homo[:, :, :3, 0]

    verts = rot_vertices.numpy()[0]

    return verts


#----------------------------------------------------------------------------
# Projection and transformation matrix helpers.
#----------------------------------------------------------------------------

def projection(x=0.1, n=1.0, f=50.0):
    return np.array([[n/x,    0,            0,              0],
                     [  0,  n/x,            0,              0],
                     [  0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                     [  0,    0,           -1,              0]]).astype(np.float32)

def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]]).astype(np.float32)

def rotate_x(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[1,  0, 0, 0],
                     [0,  c, s, 0],
                     [0, -s, c, 0],
                     [0,  0, 0, 1]]).astype(np.float32)

def rotate_y(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]]).astype(np.float32)


def project(K, R, t):

    R = R.cpu().numpy()
    t = t[0].cpu().numpy()

    # print(R.shape)
    # print(t.shape)

    translate = np.array([[1, 0, 0, -t[0]],
                          [0, 1, 0, -t[1]],
                          [0, 0, 1, -t[2]],
                          [0, 0, 0, 1]]).astype(np.float32)
    rotate =  np.array([[R[0][0], R[0][1], R[0][2], 0],
                        [R[1][0], R[1][1], R[1][2], 0],
                        [R[2][0], R[2][1], R[2][2], 0],
                        [0, 0, 0, 1]]).astype(np.float32)
    return np.matmul(rotate, translate)


#----------------------------------------------------------------------------
# Image save helper.
#----------------------------------------------------------------------------

def save_image(fn, x):
    import imageio
    x = np.rint(x * 255.0)
    x = np.clip(x, 0, 255).astype(np.uint8)
    imageio.imsave(fn, x)





