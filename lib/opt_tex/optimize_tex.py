# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import nvdiffrast.torch as dr
import matplotlib.pyplot as plt
import imageio
import numpy as np
import cv2
import os
import utils
import argparse
import glob
import torch.nn.functional as F


### image level loss
def photo_loss(imageA, imageB, mask, eps=1e-6):
    """
    l2 norm (with sqrt, to ensure backward stabililty, use eps, otherwise Nan may occur)
    Parameters:
        imageA       --torch.tensor (B, 3, H, W), range (0, 1), RGB order
        imageB       --same as imageA
    """
    loss = torch.sqrt(eps + torch.sum((imageA - imageB) ** 2, dim=1, keepdims=True)) * mask
    loss = torch.sum(loss) / torch.max(torch.sum(mask), torch.tensor(1.0).to(mask.device))
    return loss

import torch.nn as nn
class TVLoss(nn.Module):
    # [N,C,H,W]
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])  # 算出总共求了多少次差
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

def rotate_x(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[1,  0, 0],
                     [0,  c, s],
                     [0, -s, c]]).astype(np.float32)

def rotate_y(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[c,  0, -s],
                     [0,  1, 0],
                     [s, 0, c]]).astype(np.float32)

def rotate_z(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[c,  s, 0],
                     [-s,  c, 0],
                     [0, 0, 1]]).astype(np.float32)


def tensor(*args, **kwargs):
    return torch.tensor(*args, device='cuda', **kwargs)


def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

def render(glctx, mtx, pos, pos_idx, uv, uv_idx, tex, resolution, enable_mip, max_mip_level):
    pos_clip = transform_pos(mtx, pos)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])

    if enable_mip:
        texc, texd = dr.interpolate(uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
        color = dr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
    else:
        texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
        color = dr.texture(tex[None, ...], texc, filter_mode='linear')

    mask = torch.clamp(rast_out[..., -1:], 0, 1)
    color = color * mask  # Mask out background.
    return color, mask

def convert_cam(view_id, cam_ada=None, image_size = 512, loc=False, crop_dict=None):
    P = cam_ada[:16].reshape(4, 4)
    K = cam_ada[16:].reshape(3, 3)
    K[0, 0] = K[0, 0] * image_size
    K[1, 1] = K[1, 1] * image_size
    K[0, 2] = K[0, 2] * image_size
    K[1, 2] = K[1, 2] * image_size

    P_inv = np.linalg.inv(P)
    znear = 1.0
    zfar = 4.0
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    image_width = image_size
    image_height = image_size

    if loc:
        h_crop, w_crop, h_rec, w_rec = crop_dict['h_crop'], crop_dict['w_crop'], crop_dict['h_rec'], crop_dict['w_rec']
        cx = (cx - w_crop) * (image_width / w_rec)
        cy = (cy - h_crop) * (image_width / w_rec)
        fx = fx * (image_width / w_rec)
        fy = fy * (image_height / h_rec)

    right = znear * (image_width - cx) / fx
    left = -znear * cx / fx
    top = znear * (image_height - cy) / fy
    bottom = -znear * cy / fy

    proj = np.zeros((4, 4))
    proj[0, 0] = 2 * znear / (right - left)
    proj[0, 2] = (right + left) / (right - left)
    proj[1, 1] = 2 * znear / (top - bottom)
    proj[1, 2] = (top + bottom) / (top - bottom)
    proj[2, 2] = -(zfar + znear) / (zfar - znear)
    proj[2, 3] = -2 * zfar * znear / (zfar - znear)
    proj[3, 2] = -1

    flip_z = np.diag([1, 1, -1, 1])

    r_mvp = np.matmul(proj, flip_z)
    r_mvp = np.matmul(r_mvp, P_inv).astype(np.float32)

    return r_mvp


def gen_uv(vertices):
    import math
    # using right hand coordinator

    max_y = np.max(vertices[:, 1])
    min_y = np.min(vertices[:, 1])
    radius = vertices[:, 0] ** 2 + vertices[:, 2] ** 2
    radius_squared = np.max(radius)
    radius = math.sqrt(radius_squared)

    print('radius :{}'.format(radius))
    print('height :{}'.format(max_y - min_y))

    cylinder_R = radius * 1.1
    assert cylinder_R > radius
    cylinder_verts = []
    for v in vertices:
        try:
            x_proj = cylinder_R * v[0] / math.sqrt(v[0] ** 2 + v[2] ** 2)
            z_proj = cylinder_R * v[2] / math.sqrt(v[0] ** 2 + v[2] ** 2)
        except:
            print('x_proj:{}'.format(x_proj))
            print('z_proj:{}'.format(z_proj))

        proj_v = [x_proj, v[1], z_proj]
        cylinder_verts.append(proj_v)

    uv = []
    for xyzrgb in cylinder_verts:
        if xyzrgb[0] <= 0:
            tmp = -xyzrgb[2] / cylinder_R
            if abs(tmp)-1.0 > 0:
                tmp = 1.0 if tmp > 0 else -1.0
            theta = math.acos(tmp)
        else:
            theta = 2 * math.pi - math.acos(- xyzrgb[2] / cylinder_R)

        norm_u = theta / (2 * math.pi)
        norm_v = abs(xyzrgb[1] - max_y) / (max_y - min_y)
        uv.append([norm_u, 1.0 - norm_v])

    uv = np.array(uv)

    return uv

def refine_faces_uv(UVs, faces):
    """

    Args:
        UVs: np.array, size: (n, 2)
        faces: np.array, size: (m, 3), start from 1

    Returns:

    """

    # get the uv faces
    faces_uv = []
    for face in faces:
        u0 = UVs[face - 1][0, 0]
        u1 = UVs[face - 1][1, 0]
        u2 = UVs[face - 1][2, 0]
        if abs(u0 - u1) > 0.8 and abs(u0 - u2) > 0.8:
            face_uv = [face[1], face[1], face[2]]
        elif abs(u1 - u0) > 0.8 and abs(u1 - u2) > 0.8:
            face_uv = [face[0], face[0], face[2]]
        elif abs(u2 - u0) > 0.8 and abs(u2 - u1) > 0.8:
            face_uv = [face[0], face[1], face[1]]
        else:
            face_uv = face
        faces_uv.append(face_uv)
    faces_uv = np.array(faces_uv)

    return faces_uv

def get_loc(img, crop_dict):
    h_crop = crop_dict['h_crop']
    w_crop = crop_dict['w_crop']
    h_rec = crop_dict['h_rec']
    w_rec = crop_dict['w_rec']
    # print('h_crop:{}, w_crop:{}, h_rec:{}, w_rec:{}'.format(h_crop, w_crop, h_rec, w_rec))
    _, _, h_img, w_img = img.shape
    img = img[:,:, h_crop:h_crop+h_rec, w_crop:w_crop+w_rec]
    return img

class Texture(object):
    def __init__(self, img_size=512, assets_dir=None):
        model = 'data/uv_data/uv.obj'

        print('img_size:%d'%img_size)

        self.img_size = img_size
        self.type = 'human'

        self.tex_size_h = 1024
        self.tex_size_w = 1024
        if assets_dir is not None:
            self.mask_path = os.path.join(assets_dir, 'head_mask.png')
            self.head_mask = cv2.imread(self.mask_path).astype(np.float32) / 255
            self.head_mask = cv2.resize(self.head_mask, (self.tex_size_w, self.tex_size_h), interpolation=cv2.INTER_AREA)
            print('load head mask from:{}'.format(self.mask_path))



    def forward(self, args):

        tex_size_h = self.tex_size_h
        tex_size_w = self.tex_size_w
        img_size = self.img_size
        type = self.type

        dir = args['data_dir']
        mesh_path = args['mesh_uv']
        use_idx = args['use_idx']
        use_idx_list = use_idx.split(',')

        out_dir = 'tmp'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)


        #### load target image
        img_dir = os.path.join(dir, 'image')
        # img_dir = os.path.join(dir, 'color')
        img_paths = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        target_imgs = []
        for img_path in img_paths:
            img = cv2.imread(img_path)
            target_imgs.append(img)

        gt_img_path = os.path.join(dir, 'gt.png')
        # gt_img_path = os.path.join(dir, 'gt_doodle.png')

        gt_img = None
        if os.path.exists(gt_img_path):
            gt_img = cv2.imread(gt_img_path)

        gt_img_back = None
        gt_img_side = None

        # back_dir = '/data/qingyao/neuralRendering/mycode/pretrainedModel/Fantasia3D-main/3DGEN/test_images/syn_pose_prompt_format_multiview_hd_0911/6'
        case_name = dir.split('/')[-3]
        case_name = case_name[:8]+'6'+case_name[9:]
        gt_img_path_back = os.path.join(dir, 'gt_back.png')
        print('gt_img_path_back:%s'%gt_img_path_back)
        gt_img_path_side = os.path.join(dir, 'gt_side.png')

        print('use_idx_list:', use_idx_list)

        if '1' in use_idx_list and os.path.exists(gt_img_path_back):
            gt_img_back = cv2.imread(gt_img_path_back)
        if '2' in use_idx_list and os.path.exists(gt_img_path_side):
            gt_img_side = cv2.imread(gt_img_path_side)


        # load cam_ada
        cam_ada_path = os.path.join(dir, 'camera.npy')
        cam_ada = np.load(cam_ada_path)

        mesh = utils.read_obj(mesh_path)
        for k, v in mesh.items():
            print(k, v.shape)

        vert = mesh['vertices']
        tri = mesh['faces']
        tri = tri - 1 if tri.min() == 1 else tri
        uvs = None
        if 'uvs' in mesh.keys():
            uvs = mesh['uvs']
        faces_uv = None
        if 'faces_uv' in mesh.keys():
            faces_uv = mesh['faces_uv']
            faces_uv = faces_uv - 1 if faces_uv.min() == 1 else faces_uv

        if type == 'human':
            seed_img_view = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
            view_list = [0, 1, 2, 3, 4, 5, 6, -5, -4, -3, -2, -1]
            view_list_loc = view_list
            # view_list = [0]
        else:
            view_list = [-37, -32, -28, -24, -20, -17, -14, -11, -7, -3, 0, 3, 7, 11, 14, 17, 20, 24, 28, 32, 37]
            seed_img_view = [-37, -32, -28, -24, -20, -17, -14, -11, -7, -3, 0, 3, 7, 11, 14, 17, 20, 24, 28, 32, 37]

        h_crop, w_crop, h_rec, w_rec = int(img_size * 40 / 1024), int(img_size*360 / 1024), \
            int(img_size*256 / 1024), int(img_size*256 / 1024)
        # h_crop, w_crop, h_rec, w_rec = int(img_size * 40 / 1024), int(img_size * 360 / 1024), \
        #     int(img_size * 200 / 1024), int(img_size * 200 / 1024)
        crop_dict = {'h_crop': h_crop, 'h_rec': h_rec, 'w_crop': w_crop, 'w_rec': w_rec}

        # load target image
        target_img_list = []
        camera_list = []
        camera_list_loc = []
        for view_id in view_list:
            if view_id == 0 and gt_img is not None:
                target_img = gt_img
                print('use gt_img')
            elif view_id == 6 and gt_img_back is not None:
                target_img = gt_img_back
                print('use gt_img back')
            elif view_id == 3 and gt_img_side is not None:
                target_img = gt_img_side
                print('use gt_img side')
            elif view_id == -3 and gt_img_side is not None:
                target_img = gt_img_side[:,::-1,:]
                print('use gt_img side flip')
            else:
                target_img = target_imgs[seed_img_view.index(view_id)]
            target_img = cv2.resize(target_img, (img_size, img_size))[..., ::-1].astype(np.float32) / 255.0
            target_img = torch.from_numpy(target_img).cuda()
            target_img = torch.unsqueeze(target_img, dim=0).permute(0, 3, 1, 2)
            target_img_list.append(target_img)
            r_mvp = convert_cam(view_id, cam_ada[seed_img_view.index(view_id)].copy(), image_size=img_size)
            camera_list.append(r_mvp)
            r_mvp_loc = convert_cam(view_id, cam_ada[seed_img_view.index(view_id)].copy(), image_size=img_size, \
                                    loc=True, crop_dict=crop_dict)
            camera_list_loc.append(r_mvp_loc)

        ### generate uv
        uv_generated = False
        if uvs is None:
            print('generate uv...')
            uvs = gen_uv(vert)  # idx start from 1
            faces_uv = tri.copy()
            uv_generated = True

        vert_uv = uvs
        tri_uv = faces_uv - 1 if faces_uv.min() == 1 else faces_uv

        uv_mesh = {}
        uv_mesh['vertices'] = vert
        uv_mesh['faces'] = tri + 1
        uv_mesh['uvs'] = uvs
        uv_mesh['faces_uv'] = faces_uv + 1

        vtx_pos = torch.from_numpy(vert.astype(np.float32)).cuda()
        pos_idx = torch.from_numpy(tri.astype(np.int32)).cuda()
        vtx_uv = torch.from_numpy(vert_uv.astype(np.float32)).cuda()
        uv_idx = torch.from_numpy(tri_uv.astype(np.int32)).cuda()

        rows = []
        losses = []
        glctx = dr.RasterizeGLContext()
        ang = 0.0
        tex = torch.rand((tex_size_h, tex_size_w, 3), dtype=torch.float32, device='cuda', requires_grad=True)
        optim = torch.optim.Adam([tex], lr=1e-2)
        criterionTV = TVLoss()

        final_tex = None
        for i in range(int(501)):
            if i==499:
                final_tex = tex.clone()
            loss = torch.tensor(0.0).cuda()

            if i>=500:
                view_list = view_list_loc
            for view_id in range(len(view_list)):

                target_img = target_img_list[view_id]

                if i>=500:
                    r_mvp = camera_list_loc[view_id]
                    target_img = get_loc(target_img, crop_dict)
                    target_img = F.interpolate(target_img, size=(img_size, img_size), mode='bilinear',
                                               align_corners=True)
                    # if view_id>0:
                    #     target_img = target_img[:, :, :-int(0.1*img_size), :]
                    target_img = target_img[:, :, :-int(0.1 * img_size), :]

                else:
                    r_mvp = camera_list[view_id]

                pred_img, pred_mask = render(glctx, r_mvp, vtx_pos, pos_idx, vtx_uv, uv_idx, tex, img_size,
                                             enable_mip=False, max_mip_level=9)
                if i>=500:
                    pred_img = pred_img[:, :-int(0.1*img_size), :, :]
                    pred_mask = pred_mask[:, :-int(0.1*img_size), :, :]


                if torch.isnan(pred_img).any():
                    print('nan in view:{}'.format(view_id))
                    # replace nan with value in target_img
                    target_img_ = target_img.permute(0, 2, 3, 1)
                    pred_img = torch.where(torch.isnan(pred_img), target_img_, pred_img)  # continue

                rec_loss = photo_loss(pred_img.permute(0, 3, 1, 2), target_img,
                                      pred_mask.permute(0, 3, 1, 2).detach())  # 1.0
                w_view = 1.0
                if view_id != 0 and view_id != 6:
                    w_view = 0.2
                tv_loss = criterionTV(tex.unsqueeze(0).permute(0, 3, 1, 2))
                loss += rec_loss * w_view + tv_loss

            optim.zero_grad()
            loss.backward()
            optim.step()
            print('iter {}, loss :{}'.format(i, loss.item()))

            # if i % 50 == 0:
            #     utils.save_image(out_dir + '/' + ('img_%04d.png' % (i)), pred_img[0].detach().cpu().numpy())
            #     utils.save_image(out_dir + '/' + ('trg_%04d.png' % (i)),
            #                      target_img.permute(0, 2, 3, 1)[0].detach().cpu().numpy())
            #
            #     texture = np.clip(np.rint(tex.detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)[..., ::-1]
            #     texture = texture[::-1, :, :]
            #     cv2.imwrite(out_dir + '/' + ('tex_%04d.png' % i), texture)

        texture = np.clip(np.rint(tex.detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)[..., ::-1]
        texture = texture[::-1, :, :]

        if final_tex is not None:
            final_tex = np.clip(np.rint(final_tex.detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)[..., ::-1]
            final_tex = final_tex[::-1, :, :]
            final_tex = final_tex * (1-self.head_mask) + texture * self.head_mask
            texture = final_tex.copy()

        cv2.imwrite(out_dir + '/' + ('tex_final.png'), texture)

        ### save final textured mesh
        uv_mesh['texture_map'] = texture # BGR
        if uv_generated:
            print('refine_faces_uv...')
            uv_mesh['faces_uv'] = refine_faces_uv(uv_mesh['uvs'], uv_mesh['faces_uv'])

        return uv_mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--mesh_uv', default='', type=str)

    args = parser.parse_args()

    main(args)



