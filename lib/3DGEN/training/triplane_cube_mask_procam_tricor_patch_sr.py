# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from torch_utils import persistence
from training.networks_stylegan2 import Generator as StyleGAN2Backbone, MappingNetwork
from training.volumetric_rendering.renderer_cube_cor import ImportanceRenderer
from training.volumetric_rendering.ray_sampler import PatchRaySampler
import dnnlib
import torch.nn.functional as F


@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
                 z_dim,  # Input latent (Z) dimensionality.
                 c_dim,  # Conditioning label (C) dimensionality.
                 w_dim,  # Intermediate latent (W) dimensionality.
                 img_resolution,  # Output resolution.
                 img_channels,  # Number of output color channels.
                 sr_num_fp16_res=0,
                 mapping_kwargs={},  # Arguments for MappingNetwork.
                 rendering_kwargs={},
                 sr_kwargs={},
                 **synthesis_kwargs,  # Arguments for SynthesisNetwork.
                 ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.renderer = ImportanceRenderer()
        self.ray_sampler = PatchRaySampler()
        self.triplane_depth = rendering_kwargs['triplane_depth']
        self.triplane_res = rendering_kwargs['triplane_res']
        self.triplane_channel = rendering_kwargs['triplane_channel']
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=self.triplane_res, img_channels=self.triplane_channel * 3 * self.triplane_depth,
                                          mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.superresolution = dnnlib.util.construct_class_by_name(
            class_name=rendering_kwargs['superresolution_module'], channels=self.triplane_channel, img_resolution=img_resolution,
            sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.decoder = OSGDecoder(self.triplane_channel, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1),
                                       'decoder_output_dim': self.triplane_channel})
        # learn camera
        self.mapping_cam = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=3, num_ws=1, last_activation='linear',
                                        lr_multiplier=1.0, **mapping_kwargs)

        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs

        self._last_planes = None

    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi,
                                     truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def adapt_camera(self, z, c, adapt_v=None, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        # if self.rendering_kwargs['c_gen_conditioning_zero']:
        #     c = torch.zeros_like(c)
        delta_c = self.mapping_cam(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        delta_c = torch.squeeze(delta_c, 1)
        # delta_c = delta_c * 0.1 # scale for better initialization
        # index 3, 7, 11 are the translation index in extrinsic
        c_new = c.clone()
        c_new[:,3] += delta_c[:,0]
        c_new[:,7] += delta_c[:,1]
        c_new[:,11] += delta_c[:,2]

        if adapt_v is None:
            adapt_v = torch.zeros_like(c_new)
        c_new = c_new * adapt_v + c*(1-adapt_v)

        return c_new, delta_c

    def merge_color(self, feature_samples, patches):
        ### color: [batch_size, 4096, 3]
        ### patches: list
        # sqrt of colors.shape[0] is the resolution of each patch
        batch_size = feature_samples.shape[0]
        resolution = self.neural_rendering_resolution
        canvas = torch.ones((batch_size, feature_samples.shape[-1], resolution, resolution),
                             device=feature_samples.device)
        # feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H,W).contiguous()  # 4,32, 64, 64
        feature_image = feature_samples.permute(0, 2, 1)  # bs, c, 4096
        for k in range(4):
            feature_patch = feature_image[:, :,
                            k * int(feature_image.shape[-1] / 4):(k + 1) * int(feature_image.shape[-1] / 4)]
            # print(feature_image.shape)
            # reshape sample to 32*32
            patch = feature_patch.reshape(batch_size, -1, resolution // 2, resolution // 2)
            i = k // 2
            j = k % 2
            canvas[:, :, i * resolution // 2:(i + 1) * resolution // 2,
            j * resolution // 2:(j + 1) * resolution // 2] = patch
            # upsample to 64*64
            patch = F.interpolate(patch, size=(resolution, resolution), mode='bilinear', align_corners=False)
            patches.append(patch)
        return canvas, patches

    def gen_triangle_feature_mask(self, width, device):
        mask = torch.ones([width, width], device=device)
        for i in range(width):
            for j in range(width):
                if i + j < width:
                    mask[i, j] = 0
        return mask

    def recover_feature(self, patches, h_crop, w_crop, width, white_bk=False):
        bs, c, h, w = patches[0].shape
        alpha = 1 if white_bk else -1
        img = alpha* torch.ones((bs, c, 4 * h, 4 * w), device=patches[0].device)
        mask = self.gen_triangle_feature_mask(width, patches[0].device)  # h,w

        for i in range(6):
            if i == 0:
                img[:, :, h_crop:h_crop + width, w_crop:w_crop + width] = patches[i]
            elif i == 1:
                img[:, :, h_crop + 2 * width:h_crop + 3 * width, w_crop:w_crop + width] = patches[i]
            elif i == 2:
                img[:, :, max(h_crop + 3 * width, 0):min(h_crop + 4 * width, 4 * h), w_crop:w_crop + width] = patches[
                                                                                                                  i][:,
                                                                                                              :, :min(
                    h_crop + 4 * width, 4 * h) - max(h_crop + 3 * width, 0), :]
            elif i == 3:
                img[:, :, h_crop + width:h_crop + 2 * width, w_crop - width:w_crop] = patches[i] * mask +alpha*torch.ones_like(patches[i])*(1-mask)
            elif i == 4:
                img[:, :, h_crop + width:h_crop + 2 * width, w_crop:w_crop + width] = patches[i]
            elif i == 5:
                patch = patches[i - 2] * (1 - mask)+alpha*torch.ones_like(patches[i-2])*(mask)  # n,c, h, w
                img[:, :, h_crop + width:h_crop + 2 * width, w_crop + width:w_crop + 2 * width] = patch.flip(2)

        return img

    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False,
                  use_cached_backbone=False, return_planes=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)  # bs, 4,4
        intrinsics = c[:, 16:25].view(-1, 3, 3)  # bs,3,3

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create triplanes by running StyleGAN backbone
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes
        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, self.triplane_channel * self.triplane_depth, planes.shape[-2],
                             planes.shape[-1])  # bs, 3, 32, 256, 256

        feature_patches = []
        mask_patches = []

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics,
                                                       neural_rendering_resolution)  # bs, 4096,3
        N, M, _ = ray_origins.shape  # 4,4096,3
        # Perform volume rendering
        feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins,
                                                                        ray_directions,
                                                                        self.rendering_kwargs)  # channels last

        feature_samples_body = feature_samples[:, int(M/2):, :]
        feature_samples = feature_samples[:, :int(M/2), :]
        depth_samples_body = depth_samples[:, int(M/2):, :]
        depth_samples = depth_samples[:, :int(M/2), :]
        weights_samples_body = weights_samples[:, int(M/2):, :]
        weights_samples = weights_samples[:, :int(M/2), :]


        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution  # 64
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H,
                                                                 W).contiguous()  # 4,32, 64, 64
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)  # 4,1,64,64
        mask_image = weights_samples.permute(0, 2, 1).reshape(N, 1, H, W)  # bs, 1, 64, 64
        rgb_image_head = feature_image[:, :3]

        feature_patches.append(feature_image)
        mask_patches.append(mask_image)

        feature_image_body, feature_patches = self.merge_color(feature_samples_body, feature_patches) # N, C, H, W
        depth_image_body, _ = self.merge_color(depth_samples_body, []) # N, C, H, W
        mask_image_body, mask_patches = self.merge_color(weights_samples_body, mask_patches) # N, C, H, W

        h_crop_uv, w_crop_uv, width_uv = 0.02539, 0.373, 0.25
        h_crop, w_crop, width = int(h_crop_uv * self.neural_rendering_resolution * 4), int(
            w_crop_uv * self.neural_rendering_resolution * 4), int(
            width_uv * self.neural_rendering_resolution * 4)
        feature_image_full = self.recover_feature(feature_patches, h_crop, w_crop, width, self.rendering_kwargs['white_back'])

        mask_image_full = self.recover_feature(mask_patches, h_crop, w_crop, width)
        rgb_image_full = feature_image_full[:, :3]

        # # use head & body patch as raw image for Discriminator
        use_loc_patch = True
        if use_loc_patch:
            # resize feature_image
            local = F.interpolate(rgb_image_head, size=(feature_image_full.shape[2], feature_image_full.shape[3]), mode='bilinear')
            rgb_image = torch.cat([local, rgb_image_full], dim=1)
            # print('rgb_image:', rgb_image.shape)
        else:
            rgb_image = rgb_image_full

        # use fixed ws for superresolution
        ws = torch.zeros_like(ws)
        sr_image = self.superresolution(rgb_image_full, feature_image_full, ws,
                                        noise_mode=self.rendering_kwargs['superresolution_noise_mode'],
                                        **{k: synthesis_kwargs[k] for k in synthesis_kwargs.keys() if
                                           k != 'noise_mode'})

        if return_planes:
            return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, 'image_mask': mask_image_full, 'planes': planes}
        else:
            return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, 'image_mask': mask_image_full}


    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False,
               **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                          update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, self.triplane_channel * self.triplane_depth, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_prior(self, decoder, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False,
               **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                          update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, self.triplane_channel * self.triplane_depth, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixplane(self, decoder, plane_mix, coordinates):

        return self.renderer.run_model(plane_mix, decoder, coordinates, None, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False,
                     **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, self.triplane_channel * self.triplane_depth, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, adapt_v=None, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None,
                update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                          update_emas=update_emas)
        # adjust camera
        c, delta_c = self.adapt_camera(z, c, adapt_v, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

        return self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution,
                              cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone,
                              **synthesis_kwargs)

    def decode(self, planes, c, ws, neural_rendering_resolution=None, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)  # bs, 4,4
        intrinsics = c[:, 16:25].view(-1, 3, 3)  # bs,3,3

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics,
                                                       neural_rendering_resolution)  # bs, 4096,3

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape  # 4096, 4, 3

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins,
                                                                        ray_directions,
                                                                        self.rendering_kwargs)  # channels last
        # feature_samples: bs, 4096, 32

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution  # 64
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H,
                                                                 W).contiguous()  # 4,32, 64, 64
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)  # 4,1,64,64
        mask_image = weights_samples.permute(0, 2, 1).reshape(N, 1, H, W)  # bs, 1, 64, 64

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]  # 4,3,64,64
        sr_image = self.superresolution(rgb_image, feature_image, ws,
                                        noise_mode=self.rendering_kwargs['superresolution_noise_mode'],
                                        **{k: synthesis_kwargs[k] for k in synthesis_kwargs.keys() if
                                           k != 'noise_mode'})

        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, 'image_mask': mask_image}

from training.networks_stylegan2 import FullyConnectedLayer


class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'],
                                lr_multiplier=options['decoder_lr_mul'])
        )

    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N * M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:]) * (1 + 2 * 0.001) - 0.001  # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}
