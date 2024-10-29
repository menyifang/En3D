# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import torch
import nvdiffrast.torch as dr
import matplotlib.pyplot as plt
import imageio
import numpy as np
import cv2
import os
import utils
import tqdm

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

def tensor(*args, **kwargs):
    return torch.tensor(*args, device='cuda', **kwargs)

def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def render(glctx, mtx, pos, pos_idx, uv, uv_idx, tex, resolution, enable_mip, max_mip_level):
    pos_clip = transform_pos(mtx, pos)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])

    if enable_mip:
        texc, texd = dr.interpolate(uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
        color = dr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
    else:
        texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
        color = dr.texture(tex[None, ...], texc, filter_mode='linear')

    pos_idx = pos_idx.type(torch.long)
    v0 = pos[pos_idx[:, 0], :]
    v1 = pos[pos_idx[:, 1], :]
    v2 = pos[pos_idx[:, 2], :]
    face_normals = safe_normalize(torch.cross(v1 - v0, v2 - v0))
    face_normal_indices = (torch.arange(0, face_normals.shape[0], dtype=torch.int64, device='cuda')[:, None]).repeat(1, 3)
    gb_geometric_normal, _ = dr.interpolate(face_normals[None, ...], rast_out, face_normal_indices.int())
    normal = (gb_geometric_normal+1)*0.5

    mask = torch.clamp(rast_out[..., -1:], 0, 1)
    color = color * mask + (1 - mask)*torch.ones_like(color)
    normal = normal * mask + (1 - mask)*torch.ones_like(normal)
    return color, mask, normal


class Renderer(object):

    def __init__(self):
        print('Renderer init done.')

    def forward(self, mesh_path):

        out_dir = os.path.dirname(mesh_path)

        img_size = 1280

        #### load generated mesh
        mesh = utils.read_obj(mesh_path)
        tex_path = mesh_path.replace('.obj', '.png')
        print('tex_path:{}'.format(tex_path))
        tex = cv2.imread(tex_path)[::-1, :, ::-1]
        tex_ori = tex.copy()
        tex = tex.astype(np.float32) / 255.0
        tex = torch.from_numpy(tex.astype(np.float32)).cuda()

        for k, v in mesh.items():
            print(k, v.shape)
        vert = mesh['vertices']

        mesh['vertices'] = vert
        mesh['texture_map'] = tex_ori

        tri = mesh['faces']
        tri = tri - 1 if tri.min() == 1 else tri
        vert_uv = mesh['uvs']
        tri_uv = mesh['faces_uv']
        tri_uv = tri_uv - 1 if tri_uv.min() == 1 else tri_uv

        vtx_pos = torch.from_numpy(vert.astype(np.float32)).cuda()
        pos_idx = torch.from_numpy(tri.astype(np.int32)).cuda()
        vtx_uv = torch.from_numpy(vert_uv.astype(np.float32)).cuda()
        uv_idx = torch.from_numpy(tri_uv.astype(np.int32)).cuda()

        rows = []
        glctx = dr.RasterizeGLContext()

        cameras_dict = {}
        ang = 0.1
        # ang = 0
        frames = []
        frames_mix = []
        frames_color = []
        frames_color_black = []

        frame_length = 80
        step = 2 * np.pi / frame_length
        for view_id in tqdm.tqdm(range(frame_length)):

            proj = utils.projection(x=0.4, n=1.0, f=200.0)
            a_rot = np.matmul(utils.rotate_x(0.0), utils.rotate_y(ang))
            a_mv = np.matmul(utils.translate(0, 0, -2.7), a_rot)
            r_mvp = np.matmul(proj, a_mv).astype(np.float32)

            pred_img, pred_mask, normal = render(glctx, r_mvp, vtx_pos, pos_idx, vtx_uv, uv_idx, tex, img_size,
                                                 enable_mip=False,
                                                 max_mip_level=9)

            pred_img_black = pred_img * pred_mask + (1 - pred_mask)*torch.zeros_like(pred_img)

            color = np.clip(np.rint(pred_img[0].detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)[::-1, :, :]
            normal = np.clip(np.rint(normal[0].detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)[::-1, :, :]
            # crop normal
            nw = int(img_size*0.7)
            normal = normal[:, int(img_size/2)-int(nw/2):int(img_size/2)-int(nw/2)+nw, :]
            # crop color
            color = color[:, int(img_size/2)-int(nw/2):int(img_size/2)-int(nw/2)+nw, :]

            # color black
            color_balck = np.clip(np.rint(pred_img_black[0].detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)[::-1, :, :]
            color_balck = color_balck[:, int(img_size/2)-int(nw/2):int(img_size/2)-int(nw/2)+nw, :]

            combine = np.column_stack((color, normal))
            # combine = color.copy()
            if view_id == 0:
                cv2.imwrite(os.path.join(os.path.dirname(out_dir), 'render_front.png'), combine[:, :, ::-1])
            elif view_id == 39:
                cv2.imwrite(os.path.join(os.path.dirname(out_dir), 'render_back.png'), combine[:, :, ::-1])
            elif view_id == 19:
                cv2.imwrite(os.path.join(os.path.dirname(out_dir), 'render_side.png'), combine[:, :, ::-1])

            # writer.append_data(combine)
            frames.append(combine)
            frames_color.append(color)
            frames_color_black.append(color_balck)

            combine2 = color.copy()
            h, w, c = combine2.shape
            combine2[:, int(w/2):,:] = normal[:, int(w/2):,:]
            frames_mix.append(combine2)
            ang = ang + step

        imageio.mimsave(os.path.join(os.path.dirname(out_dir), 'render.mp4'), frames, fps=30, quality=8, macro_block_size=1)


def main(args):
    tex_obj_path = args.final_path
    renderer = Renderer()
    renderer.forward(tex_obj_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--final_path', default=None, type=str)

    args = parser.parse_args()


    main(args)
