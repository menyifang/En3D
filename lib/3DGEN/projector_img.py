# Copyright (c) Alibaba Group.  All rights reserved.
"""Project given image to the latent space of pretrained network pickle."""
import copy
import os
from time import perf_counter
import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import dnnlib
import legacy
from gen_cameras_syn import Converter
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
import trimesh
import skimage.measure
import json

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

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size


def project(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    c: torch.Tensor,
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    optimize_noise             = False,
    verbose                    = False,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    # camera_lookat_point = torch.tensor([0, 0, 0.0], device=device)
    # cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point, radius=2.7, device=device)
    # intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    # c_samples = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    c_samples = torch.zeros_like(c)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), c_samples.repeat(w_avg_samples,1))  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    url = 'models/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32) / 255.0 * 2 - 1
    target_images_perc = (target_images + 1) * (255/2)
    if target_images_perc.shape[2] > 256:
        target_images_perc = F.interpolate(target_images_perc, size=(256, 256), mode='area')
    target_features = vgg16(target_images_perc, resize_images=False, return_lpips=True)

    print('num_ws:', G.backbone.mapping.num_ws) #14
    w_avg = torch.tensor(w_avg, dtype=torch.float32, device=device).repeat(1, G.backbone.mapping.num_ws, 1)
    w_opt = w_avg.detach().clone()
    w_opt.requires_grad = True
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device="cpu")
    if optimize_noise:
        optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)
    else:
        optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    if optimize_noise:
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

    save_dir = 'tmp'
    os.makedirs(save_dir, exist_ok=True)
    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = w_opt + w_noise
        # print('ws.shape:', ws.shape)
        synth_images = G.synthesis(ws, c=c, noise_mode='const')['image']

        #### visualize synth_images
        if step %100 == 0 or step== num_steps-1:
            print('synth_images.shape: ', synth_images.shape)
            vis_syn = synth_images[0].detach().cpu().numpy().transpose(1, 2, 0)
            vis_syn = (vis_syn + 1) * 0.5
            vis_syn = np.clip(vis_syn, 0, 1) * 255
            vis_syn = vis_syn.astype(np.uint8)
            import cv2
            outpath = os.path.join(save_dir, 'synth_image_%d.png' % step)
            cv2.imwrite(outpath, vis_syn[:, :, ::-1])



        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images_perc = (synth_images + 1) * (255/2)
        if synth_images_perc.shape[2] > 256:
            synth_images_perc = F.interpolate(synth_images_perc, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images_perc, resize_images=False, return_lpips=True)
        perc_loss = (target_features - synth_features).square().sum(1).mean()

        mse_loss = (target_images - synth_images).square().mean()

        w_norm_loss = (w_opt-w_avg).square().mean()

        # Noise regularization.
        reg_loss = 0.0
        if optimize_noise:
            for v in noise_bufs.values():
                noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
        loss = 0.1 * mse_loss + perc_loss + 1.0 * w_norm_loss +  reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step: {step+1:>4d}/{num_steps} mse: {mse_loss:<4.2f} perc: {perc_loss:<4.2f} w_norm: {w_norm_loss:<4.2f}  noise: {float(reg_loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach().cpu()[0]

        # Normalize noise.
        if optimize_noise:
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

    if w_out.shape[1] == 1:
        w_out = w_out.repeat([1, G.mapping.num_ws, 1])

    return w_out, c


def project_pti(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    w_pivot: torch.Tensor,
    c: torch.Tensor,
    *,
    num_steps                  = 1000,
    initial_learning_rate      = 3e-4,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    verbose                    = False,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).train().requires_grad_(True).to(device) # type: ignore

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32) / 255.0 * 2 - 1
    target_images_perc = (target_images + 1) * (255/2)
    if target_images_perc.shape[2] > 256:
        target_images_perc = F.interpolate(target_images_perc, size=(256, 256), mode='area')
    target_features = vgg16(target_images_perc, resize_images=False, return_lpips=True)

    w_pivot = w_pivot.to(device).detach()
    optimizer = torch.optim.Adam(G.parameters(), betas=(0.9, 0.999), lr=initial_learning_rate)

    out_params = []

    for step in range(num_steps):

        # Synth images from opt_w.
        synth_images = G.synthesis(w_pivot, c=c, noise_mode='const')['image']

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images_perc = (synth_images + 1) * (255/2)
        if synth_images_perc.shape[2] > 256:
            synth_images_perc = F.interpolate(synth_images_perc, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images_perc, resize_images=False, return_lpips=True)
        perc_loss = (target_features - synth_features).square().sum(1).mean()

        mse_loss = (target_images - synth_images).square().mean()

        loss = 0.1 * mse_loss + perc_loss

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step: {step+1:>4d}/{num_steps} mse: {mse_loss:<4.2f} perc: {perc_loss:<4.2f}')

        if step == num_steps - 1 or step % 10 == 0:
            out_params.append(copy.deepcopy(G).eval().requires_grad_(False).cpu())

    return out_params

def generate_images(G, camera_params_dict,
    latent: str,
    output: str,
    truncation_psi=0.7,
    truncation_cutoff=14,
    grid=(1,1),
    num_keyframes=None,
    w_frames=240,
    reload_modules=False,
    cfg='human',
    image_mode='image',
    sampling_multiplier=1,
    nrr=None,
    shapes=True,
    shape_res=512,
    interpolate=True,
    fov_deg=18.837,
):
    """Render a latent vector interpolation video.

    """

    converter = Converter()

    device = torch.device('cuda')

    if truncation_cutoff == 0:
        truncation_psi = 1.0 # truncation cutoff of 0 means no truncation anyways
    if truncation_psi == 1.0:
        truncation_cutoff = 14 # no truncation so doesn't matter where we cutoff

    ws = torch.tensor(np.load(latent)['w']).to(device)
    # gen_interp_video(G=G, mp4=output, ws=ws, bitrate='100M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames, psi=truncation_psi, truncation_cutoff=truncation_cutoff, cfg=cfg, image_mode=image_mode, gen_shapes=shapes, device=device)
    if not os.path.exists(output):
        os.makedirs(output)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    if cfg == "Head":
        fx = 1.3077
    elif cfg == "human":
        fx = 1.09375

    intrinsics[0][0] = fx
    intrinsics[1][1] = fx

    if cfg == 'human':
        use_v = [0, 1, 2, 3, 4, 5, 6, -6, -5, -4, -3, -2, -1]
        use_v_adpat = []
    else:
        use_v_adpat = [11, 14, 17, 20, 24, 28, 32, 37]
        use_v = [0, 11, 14, 17, 20, 24, 28, 32, 37]

    camera_params_list = []
    adapt_v_list = []
    for key in camera_params_dict.keys():#[-14, 0, 14]
        print('key', key)
        if key in use_v:
            cam = torch.tensor(camera_params_dict[key], device=device).reshape(-1, 25)
            camera_params_list.append(cam)
            adapt_v = torch.zeros_like(cam)
            if abs(key) in use_v_adpat:
                adapt_v[:, :] = 1
            adapt_v_list.append(adapt_v)

    ## generate mesh and renderings
    imgs = []
    idx = 0
    cam_new = []
    print('render %d view images' % len(camera_params_list))
    for camera_params in camera_params_list:
        adapt_v = adapt_v_list[idx]
        z = torch.from_numpy(np.random.RandomState(0).randn(1, G.z_dim)).to(device)
        camera_params, cam_delta = G.adapt_camera(z, camera_params, adapt_v, truncation_psi=truncation_psi,
                                                  truncation_cutoff=truncation_cutoff)
        cam_new.append(camera_params)
        img = G.synthesis(ws, camera_params)['image']

        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        imgs.append(img)
        idx += 1

    img = torch.cat(imgs, dim=2)
    seed = 0
    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{output}/seed{seed:04d}.png')
    # save camera params
    cam_new = torch.cat(cam_new, dim=0)
    np.save(f'{output}/seed{seed:04d}.npy', cam_new.cpu().numpy())

    if shapes:
        # extract a shape.mrc with marching cubes. You can view the .mrc file using ChimeraX from UCSF.
        max_batch = 1000000

        samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0],
                                                           cube_length=G.rendering_kwargs[
                                                                           'box_warp'] * 1)  # .reshape(1, -1, 3)
        samples = samples.to(z.device)
        sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
        transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
        transformed_ray_directions_expanded[..., -1] = -1

        head = 0
        with tqdm(total=samples.shape[1]) as pbar:
            with torch.no_grad():
                while head < samples.shape[1]:
                    torch.manual_seed(0)
                    sigma = G.sample_mixed(samples[:, head:head + max_batch],
                                           transformed_ray_directions_expanded[:, :samples.shape[1] - head],
                                           ws, truncation_psi=truncation_psi, noise_mode='const')['sigma']
                    sigmas[:, head:head + max_batch] = sigma
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
        outpath = os.path.join(output, f'seed{seed:04d}.ply')
        mesh.export(outpath)
        mesh.export(outpath.replace('.ply', '.obj'))
        print('mesh save to %s' % outpath)

    # format multi-view data
    outdir_mv = os.path.join(output + '_multiview', f'seed{seed:04d}')
    args = {'data_dir': output, 'seed': seed, 'out_dir': outdir_mv}
    converter.convert_data(args)

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--camera', 'camera_file', help='camera_file json filename', required=True)
@click.option('--target', 'target_fname',       help='Target image file to project to', required=True, metavar='FILE|DIR')
@click.option('--idx',                    help='index from dataset', type=int, default=0,  metavar='FILE|DIR')
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=500, show_default=True)
@click.option('--num-steps-pti',          help='Number of optimization steps for pivot tuning', type=int, default=500, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=666, show_default=True)
@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
@click.option('--fps',                    help='Frames per second of final video', default=30, show_default=True)
@click.option('--shapes', type=bool, help='Gen shapes for shape interpolation', default=False, show_default=True)

def run_projection(
    network_pkl: str,
    camera_file: str,
    target_fname: str,
    idx: int,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int,
    num_steps_pti: int,
    fps: int,
    shapes: bool,
):
    """Project given image to the latent space of pretrained network pickle.

    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Render debug output: optional video and projected image and W vector.
    # outdir = os.path.join(outdir, os.path.basename(network_pkl), str(idx))
    os.makedirs(outdir, exist_ok=True)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        network_data = legacy.load_network_pkl(fp)
        G = network_data['G_ema'].requires_grad_(False).to(device) # type: ignore
    
    # G.rendering_kwargs["ray_start"] = 2.35

    # load target RGBA image
    target_pil = PIL.Image.open(target_fname).convert('RGBA')
    # resize target image to 256x256
    target_pil = target_pil.resize((512, 512), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)[..., :3]

    # load target back RGBA image
    target_fname_back = target_fname[:-4] + '_back.png'
    if os.path.exists(target_fname_back):
        target_pil_back = PIL.Image.open(target_fname_back).convert('RGBA')
        target_pil_back = target_pil_back.resize((512, 512), PIL.Image.LANCZOS)
        target_pil_back.save(f'{outdir}/target_back.png')

    # load target side RGBA image
    target_fname_side = target_fname[:-4] + '_side.png'
    if os.path.exists(target_fname_side):
        target_pil_side = PIL.Image.open(target_fname_side).convert('RGBA')
        target_pil_side = target_pil_side.resize((512, 512), PIL.Image.LANCZOS)
        target_pil_side.save(f'{outdir}/target_side.png')

    # load camera
    with open(f'models/camera.json', 'r') as f:
        camera_params = json.load(f)
    camera_params_dict = {}
    for key in camera_params.keys():
        if '-' in key:
            new_key = -int(key[1:])
        else:
            new_key = int(key)
        camera_params_dict[new_key] = camera_params[key]
    print(camera_params_dict.keys())

    c = torch.tensor(camera_params_dict[0], device=device).reshape(-1, 25) # real view id here

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps, c = project(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        c=c,
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

    # Save final projected frame and W vector.
    target_pil.save(f'{outdir}/target.png')

    projected_w = projected_w_steps[-1]
    out_latent = f'{outdir}/projected_w.npz'
    np.savez(out_latent, w=projected_w.unsqueeze(0).cpu().numpy())

    # synthesize samples
    output = os.path.join(outdir, 'PTI_render')
    generate_images(G, camera_params_dict, out_latent, output, truncation_psi = 0.7, truncation_cutoff = 14, shapes=True)



#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------