# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import re
from typing import List, Union
import json
import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import skimage.measure
import trimesh
import legacy
import math
from torch_utils import misc
from training.triplane_cube_mask_procam_tricor_patch import TriPlaneGenerator
from gen_cameras_syn import Converter
#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------

def FOV_to_intrinsics(fov_degrees, device='cpu'):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """
    focal_length = float(1 / (math.tan(fov_degrees * 3.14159 / 360) * 1.414))
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    return intrinsics

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--camera', 'camera_file', help='camera_file json filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--planes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--type', type=str, required=False, default='head')

def main(
    network_pkl: str,
    camera_file: str,
    seeds: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    shapes: bool,
    planes: bool,
    shape_res: int,
    fov_deg: float,
    reload_modules: bool,
    type: str,
):
    """Generate images and features using pretrained network pickle"""

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new

    converter = Converter()

    os.makedirs(outdir, exist_ok=True)

    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    if type == 'human':
        fx = 1.09375
    else:
        fx = 1.3077
    intrinsics[0][0] = fx
    intrinsics[1][1] = fx

    # load the camera
    with open(camera_file, 'r') as f:
        camera_params = json.load(f)
    camera_params_dict = {}
    for key in camera_params.keys():
        if '-' in key:
            new_key = -int(key[1:])
        else:
            new_key = int(key)
        camera_params_dict[new_key] = camera_params[key]

    # convert dict to list
    camera_params_list = []
    adapt_v_list = []
    if type== 'human':
        use_v = [0, 1, 2, 3, 4, 5, 6, -6, -5, -4, -3, -2, -1]
        use_v_adpat = []

    for key in camera_params_dict.keys():
        if key in use_v:
            cam = torch.tensor(camera_params_dict[key], device=device).reshape(-1, 25)
            camera_params_list.append(cam)
            adapt_v = torch.zeros_like(cam)
            if abs(key) in use_v_adpat:
                adapt_v[:, :] = 1
            adapt_v_list.append(adapt_v)


    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating 360 renderings for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

        idx = 0
        imgs = []
        cam = []
        triplane = None
        for camera_params in camera_params_list:
            if planes and idx == 0:
                return_planes = True
            else:
                return_planes = False
            conditioning_params = torch.zeros_like(camera_params)
            ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            camera_params, cam_delta = G.adapt_camera(z, camera_params, adapt_v, truncation_psi=truncation_psi,
                                           truncation_cutoff=truncation_cutoff)

            cam.append(camera_params)
            out = G.synthesis(ws, camera_params, return_planes=return_planes, noise_mode='const')
            img = out['image']
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            imgs.append(img)
            idx += 1

        img = torch.cat(imgs, dim=2)
        # save multi-view images
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')

        # save generated mesh
        if shapes:
            max_batch=1000000
            samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'] * 1)
            samples = samples.to(z.device)
            sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
            transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
            transformed_ray_directions_expanded[..., -1] = -1

            head = 0
            with tqdm(total = samples.shape[1]) as pbar:
                with torch.no_grad():
                    while head < samples.shape[1]:
                        torch.manual_seed(0)
                        sigma = G.sample(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, noise_mode='const')['sigma']
                        sigmas[:, head:head+max_batch] = sigma
                        head += max_batch
                        pbar.update(max_batch)

            sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
            verts, faces, normals, values = skimage.measure.marching_cubes(sigmas, level=10, spacing=[1] * 3)
            verts = verts * G.rendering_kwargs['box_warp'] / shape_res - 1
            # clean mesh
            mesh_lst = trimesh.Trimesh(verts, faces)
            mesh_lst = mesh_lst.split(only_watertight=False)
            comp_num = [mesh.vertices.shape[0] for mesh in mesh_lst]
            mesh_clean = mesh_lst[comp_num.index(max(comp_num))]
            verts = mesh_clean.vertices
            faces = mesh_clean.faces  # start from 0
            mesh = trimesh.Trimesh(verts, faces[..., ::-1])
            outpath = os.path.join(outdir, f'seed{seed:04d}.obj')
            mesh.export(outpath)
            print('mesh save to %s' % outpath)



#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
