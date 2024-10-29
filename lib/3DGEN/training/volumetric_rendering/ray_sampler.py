# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
The ray sampler is a module that takes in camera matrices and resolution and batches of rays.
Expects cam2world matrices that use the OpenCV camera coordinate system conventions.
"""

import torch

class RaySampler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ray_origins_h, self.ray_directions, self.depths, self.image_coords, self.rendering_options = None, None, None, None, None


    def forward(self, cam2world_matrix, intrinsics, resolution):
        """
        Create batches of rays and return origins and directions.

        cam2world_matrix: (N, 4, 4)
        intrinsics: (N, 3, 3)
        resolution: int

        ray_origins: (N, M, 3)
        ray_dirs: (N, M, 2)
        """
        N, M = cam2world_matrix.shape[0], resolution**2
        cam_locs_world = cam2world_matrix[:, :3, 3]
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        sk = intrinsics[:, 0, 1]

        uv = torch.stack(torch.meshgrid(torch.arange(resolution, dtype=torch.float32, device=cam2world_matrix.device), torch.arange(resolution, dtype=torch.float32, device=cam2world_matrix.device), indexing='ij')) * (1./resolution) + (0.5/resolution)
        uv = uv.flip(0).reshape(2, -1).transpose(1, 0)
        uv = uv.unsqueeze(0).repeat(cam2world_matrix.shape[0], 1, 1)

        x_cam = uv[:, :, 0].view(N, -1)
        y_cam = uv[:, :, 1].view(N, -1)
        z_cam = torch.ones((N, M), device=cam2world_matrix.device)

        x_lift = (x_cam - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y_cam/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z_cam
        y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

        cam_rel_points = torch.stack((x_lift, y_lift, z_cam, torch.ones_like(z_cam)), dim=-1)

        world_rel_points = torch.bmm(cam2world_matrix, cam_rel_points.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3]

        ray_dirs = world_rel_points - cam_locs_world[:, None, :]
        ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2)

        ray_origins = cam_locs_world.unsqueeze(1).repeat(1, ray_dirs.shape[1], 1)

        return ray_origins, ray_dirs

class CompRaySampler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ray_origins_h, self.ray_directions, self.depths, self.image_coords, self.rendering_options = None, None, None, None, None


    def forward(self, cam2world_matrix, intrinsics, resolution, loc_info=None):
        """
        Create batches of rays and return origins and directions.

        cam2world_matrix: (N, 4, 4)
        intrinsics: (N, 3, 3)
        resolution: int

        ray_origins: (N, M, 3)
        ray_dirs: (N, M, 2)
        """
        N, M = cam2world_matrix.shape[0], resolution**2
        cam_locs_world = cam2world_matrix[:, :3, 3]
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        sk = intrinsics[:, 0, 1]

        if loc_info is not None:
            h_crop, w_crop, width = loc_info['h_crop'], loc_info['w_crop'], loc_info['width']
        else:
            h_crop, w_crop, width = 0, 0, 1
        h_pixels = torch.arange(resolution, dtype=torch.float32, device=cam2world_matrix.device) * (1. / resolution) * width + h_crop
        w_pixels = torch.arange(resolution, dtype=torch.float32, device=cam2world_matrix.device) * (1. / resolution) * width + w_crop
        uv = torch.stack(torch.meshgrid(h_pixels, w_pixels, indexing='ij'))

        # uv = torch.stack(*) * (1./resolution) + (0.5/resolution)
        uv = uv.flip(0).reshape(2, -1).transpose(1, 0)
        uv = uv.unsqueeze(0).repeat(cam2world_matrix.shape[0], 1, 1)

        x_cam = uv[:, :, 0].view(N, -1)
        y_cam = uv[:, :, 1].view(N, -1)
        z_cam = torch.ones((N, M), device=cam2world_matrix.device)

        x_lift = (x_cam - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y_cam/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z_cam
        y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

        cam_rel_points = torch.stack((x_lift, y_lift, z_cam, torch.ones_like(z_cam)), dim=-1)

        world_rel_points = torch.bmm(cam2world_matrix, cam_rel_points.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3]

        ray_dirs = world_rel_points - cam_locs_world[:, None, :]
        ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2)

        ray_origins = cam_locs_world.unsqueeze(1).repeat(1, ray_dirs.shape[1], 1)

        return ray_origins, ray_dirs

def get_triangle_uv_mask(width):
    # init uv mask
    uv_mask = torch.ones(int(width**2))
    # uv_mask = torch.ones([width**2])
    for i in range(width):
        for j in range(width):
            if i+j < width:
                uv_mask[i*width+j] = 0
    return uv_mask
class PatchRaySampler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ray_origins_h, self.ray_directions, self.depths, self.image_coords, self.rendering_options = None, None, None, None, None

    def gen_patch_pixel_uv(self, resolution, h_crop, w_crop, width, device, mode='head'):
        if mode == 'head':
            h_pixels = torch.arange(resolution, dtype=torch.float32, device=device) * (1. / resolution) * width + h_crop
            w_pixels = torch.arange(resolution, dtype=torch.float32, device=device) * (1. / resolution) * width + w_crop
            uv = torch.stack(torch.meshgrid(h_pixels, w_pixels, indexing='ij'))
            uv = uv.flip(0).reshape(2, -1).transpose(1, 0)
        else:
            resolution = resolution // 2

            uv_list = []
            for i in range(1, 6):
                if i == 1:
                    h_pixel = torch.arange(resolution, dtype=torch.float32, device=device) * (
                            1. / resolution) * width + h_crop + 2 * width
                    w_pixel = torch.arange(resolution, dtype=torch.float32, device=device) * (1. / resolution) * width + w_crop
                elif i == 2:
                    h_pixel = torch.arange(resolution, dtype=torch.float32, device=device) * (
                            1. / resolution) * width + h_crop + 3 * width
                    h_pixel[h_pixel >= 1] = 1 - 0.5 / resolution
                    w_pixel = torch.arange(resolution, dtype=torch.float32, device=device) * (1. / resolution) * width + w_crop
                elif i == 3:
                    h_pixel = torch.arange(resolution, dtype=torch.float32, device=device) * (
                            1. / resolution) * width + h_crop + width
                    w_pixel = torch.arange(resolution, dtype=torch.float32, device=device) * (1. / resolution) * width + w_crop - width
                elif i == 4:
                    h_pixel = torch.arange(resolution, dtype=torch.float32, device=device) * (
                            1. / resolution) * width + h_crop + width
                    w_pixel = torch.arange(resolution, dtype=torch.float32, device=device) * (1. / resolution) * width + w_crop
                elif i == 5:
                    h_pixel = torch.arange(resolution, dtype=torch.float32, device=device) * (
                            1. / resolution) * width + h_crop + width
                    # reverse h_pixel
                    h_pixel = h_pixel.flip(0)
                    w_pixel = torch.arange(resolution, dtype=torch.float32, device=device) * (1. / resolution) * width + w_crop + width
                    uv = torch.stack(torch.meshgrid(h_pixel, w_pixel, indexing='ij'))
                    uv = uv.flip(0).reshape(2, -1).transpose(1, 0)
                    uv_mask = get_triangle_uv_mask(resolution)
                    mask = (uv_mask == 0)
                    uv_list[i - 3][mask, :] = uv[mask, :]

                if i != 5:
                    uv = torch.stack(torch.meshgrid(h_pixel, w_pixel, indexing='ij'))
                    uv = uv.flip(0).reshape(2, -1).transpose(1, 0)
                    # print('uv:', uv.shape)
                    uv_list.append(uv)

            uv = torch.cat(uv_list, dim=0)

        return uv

    def forward(self, cam2world_matrix, intrinsics, resolution, mode='head'):
        """
        Create batches of rays and return origins and directions.

        cam2world_matrix: (N, 4, 4)
        intrinsics: (N, 3, 3)
        resolution: int

        ray_origins: (N, M, 3)
        ray_dirs: (N, M, 2)
        """
        N, M = cam2world_matrix.shape[0], resolution**2
        cam_locs_world = cam2world_matrix[:, :3, 3]
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        sk = intrinsics[:, 0, 1]

        # uv = torch.stack(torch.meshgrid(torch.arange(resolution, dtype=torch.float32, device=cam2world_matrix.device), torch.arange(resolution, dtype=torch.float32, device=cam2world_matrix.device), indexing='ij')) * (1./resolution) + (0.5/resolution)
        # uv = uv.flip(0).reshape(2, -1).transpose(1, 0)

        h_crop_uv, w_crop_uv, width_uv = 0.02539, 0.373, 0.25
        # h_crop_uv, w_crop_uv, width_uv = 0, 0, 1
        uv = self.gen_patch_pixel_uv(resolution, h_crop_uv, w_crop_uv, width_uv, device=cam2world_matrix.device, mode='head')
        uv_body = self.gen_patch_pixel_uv(resolution, h_crop_uv, w_crop_uv, width_uv, device=cam2world_matrix.device, mode='body')
        uv = torch.cat((uv, uv_body), dim=0)
        M *=2

        uv = uv.unsqueeze(0).repeat(cam2world_matrix.shape[0], 1, 1)

        x_cam = uv[:, :, 0].view(N, -1)
        y_cam = uv[:, :, 1].view(N, -1)
        z_cam = torch.ones((N, M), device=cam2world_matrix.device)

        x_lift = (x_cam - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y_cam/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z_cam
        y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

        cam_rel_points = torch.stack((x_lift, y_lift, z_cam, torch.ones_like(z_cam)), dim=-1)

        world_rel_points = torch.bmm(cam2world_matrix, cam_rel_points.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3]

        ray_dirs = world_rel_points - cam_locs_world[:, None, :]
        ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2)

        ray_origins = cam_locs_world.unsqueeze(1).repeat(1, ray_dirs.shape[1], 1)

        return ray_origins, ray_dirs