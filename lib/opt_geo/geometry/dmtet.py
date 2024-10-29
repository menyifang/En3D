# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import numpy as np
import torch

from render import mesh
from render import render
from render import util
from geometry.dmtet_network import Decoder
from render import regularizer
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd 

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

###############################################################################
# Regularizer
###############################################################################

def sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1,2)
    mask = torch.sign(sdf_f1x6x2[...,0]) != torch.sign(sdf_f1x6x2[...,1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,0], (sdf_f1x6x2[...,1] > 0).float()) + \
            torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,1], (sdf_f1x6x2[...,0] > 0).float())
    return sdf_diff

###############################################################################
# Marching tetrahedrons implementation (differentiable), adapted from
# https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py
###############################################################################

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None
    
class DMTet:
    def __init__(self):
        self.triangle_table = torch.tensor([   
                [-1, -1, -1, -1, -1, -1],     
                [ 1,  0,  2, -1, -1, -1],     
                [ 4,  0,  3, -1, -1, -1],
                [ 1,  4,  2,  1,  3,  4],
                [ 3,  1,  5, -1, -1, -1],
                [ 2,  3,  0,  2,  5,  3],     
                [ 1,  4,  0,  1,  5,  4],
                [ 4,  2,  5, -1, -1, -1],
                [ 4,  5,  2, -1, -1, -1],
                [ 4,  1,  0,  4,  5,  1],
                [ 3,  2,  0,  3,  5,  2],
                [ 1,  3,  5, -1, -1, -1],
                [ 4,  1,  2,  4,  3,  1],
                [ 3,  0,  4, -1, -1, -1],
                [ 2,  0,  1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1]
                ], dtype=torch.long).cuda()   
        self.num_triangles_table = torch.tensor([0,1,1,2,1,2,2,1,1,2,2,1,2,1,1,0], dtype=torch.long, device='cuda') #三角面片数量表，一共有16种类型的四面体，每一种四面体都对应要提取出的三角面片数量
        self.base_tet_edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype=torch.long, device='cuda') #基本的边，每一个数字是顶点索引，总共有12条边

    ###############################################################################
    # Utility functions
    ###############################################################################

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:,0] > edges_ex2[:,1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)      
            b = torch.gather(input=edges_ex2, index=1-order, dim=1)  

        return torch.stack([a, b],-1)

    def map_uv(self, faces, face_gidx, max_idx):
        N = int(np.ceil(np.sqrt((max_idx+1)//2)))
        tex_y, tex_x = torch.meshgrid(
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda")
        )

        pad = 0.9 / N

        uvs = torch.stack([
            tex_x      , tex_y,
            tex_x + pad, tex_y,
            tex_x + pad, tex_y + pad,
            tex_x      , tex_y + pad
        ], dim=-1).view(-1, 2)

        def _idx(tet_idx, N):
            x = tet_idx % N
            y = torch.div(tet_idx, N, rounding_mode='trunc')
            return y * N + x

        tet_idx = _idx(torch.div(face_gidx, 2, rounding_mode='trunc'), N)
        tri_idx = face_gidx % 2

        uv_idx = torch.stack((
            tet_idx * 4, tet_idx * 4 + tri_idx + 1, tet_idx * 4 + tri_idx + 2
        ), dim = -1). view(-1, 3)

        return uvs, uv_idx

    ###############################################################################
    # Marching tets implementation
    ###############################################################################

    def __call__(self, pos_nx3, sdf_n, tet_fx4):
        # print("pos_nx3", pos_nx3.shape)
        # print("sdf_n", sdf_n.shape)
        # print("tet_fx4", tet_fx4.shape)
        with torch.no_grad():
            occ_n = sdf_n > 0 
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1,4) 
            occ_sum = torch.sum(occ_fx4, -1) 
            valid_tets = (occ_sum>0) & (occ_sum<4) 
            occ_sum = occ_sum[valid_tets] 
            all_edges = tet_fx4[valid_tets][:,self.base_tet_edges].reshape(-1,2) 
            all_edges = self.sort_edges(all_edges) 
            unique_edges, idx_map = torch.unique(all_edges,dim=0, return_inverse=True)    
            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1,2).sum(-1) == 1 
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device="cuda") * -1 
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long,device="cuda") 
            idx_map = mapping[idx_map] 
            interp_v = unique_edges[mask_edges] 
            
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1,2,3) 
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1,2,1) 
        edges_to_interp_sdf[:,-1] *= -1

        denominator = edges_to_interp_sdf.sum(1,keepdim = True) 
        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1])/denominator 
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1) 
        idx_map = idx_map.reshape(-1,6)
        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device="cuda")) 
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1) 
        num_triangles = self.num_triangles_table[tetindex] 

        faces = torch.cat((
            torch.gather(input=idx_map[num_triangles == 1], dim=1, index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1,3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1, index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1,3),
        ), dim=0)

        return verts, faces

###############################################################################
#  Geometry interface
###############################################################################

class DMTetGeometry(torch.nn.Module):
    def __init__(self, grid_res, scale, FLAGS):
        super(DMTetGeometry, self).__init__()

        self.FLAGS         = FLAGS
        self.grid_res      = grid_res
        self.marching_tets = DMTet()
 
        # self.decoder.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / gradient_scaling, ))
        tets = np.load('../../models/{}_tets.npz'.format(self.grid_res))
        self.verts    =  torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda') * scale
        # self.verts = self.verts*0.5
        # self.verts[..., 1] += 0.5

        # self.verts[..., 0] *= 0.8
        # self.verts[..., 2] *= 0.8
        self.grid_scale = scale
        # print('tet verts', self.verts.shape) #[2158402, 3]
        # print("tet grid min/max", torch.min(self.verts).item(), torch.max(self.verts).item())
        self.decoder = Decoder(multires=0 , AABB= self.getAABB(), mesh_scale= scale)
        self.indices  = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')
        self.generate_edges()

    def generate_edges(self):
        with torch.no_grad():
            edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype = torch.long, device = "cuda")
            all_edges = self.indices[:,edges].reshape(-1,2) 
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)

    def get_sdf_gradient(self):
        num_samples = 5000
        sample_points = (torch.rand(num_samples, 3, device=self.verts.device) - 0.5) * self.grid_scale
        mesh_verts = self.mesh_verts.detach() + (torch.rand_like(self.mesh_verts) -0.5) * 0.1 * self.grid_scale
        rand_idx = torch.randperm(len(mesh_verts), device=mesh_verts.device)[:5000]
        mesh_verts = mesh_verts[rand_idx]
        sample_points = torch.cat([sample_points, mesh_verts], 0)
        sample_points.requires_grad = True
        pred = self.decoder(sample_points)
        y , _ =  pred[:, 0], pred[:, 1:]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        try:
            gradients = torch.autograd.grad(
                outputs=[y],
                inputs=sample_points,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        except RuntimeError:  # For validation, we have disabled gradient calculation.
            return torch.zeros_like(sample_points)
        return gradients


    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values

    def getMesh(self, material):
        pred= self.decoder(self.verts)
        
        self.sdf , self.deform =  pred[:, 0], pred[:, 1:]  
        v_deformed = self.verts + 1 / (self.grid_res ) * torch.tanh(self.deform)
        verts, faces = self.marching_tets(v_deformed, self.sdf, self.indices)
        self.mesh_verts = verts

        imesh = mesh.Mesh(verts, faces, material=material)
        imesh = mesh.auto_normals(imesh)
        return imesh

    def getMesh_possion(self, material):
        pred = self.decoder(self.verts)

        self.sdf, self.deform = pred[:, 0], pred[:, 1:]
        v_deformed = self.verts + 1 / (self.grid_res) * torch.tanh(self.deform)
        verts, faces = self.marching_tets(v_deformed, self.sdf, self.indices)

        import trimesh
        mesh_mid = trimesh.Trimesh(verts.cpu().numpy(), faces.cpu().numpy(), process=False, maintains_order=True)
        from pypoisson import poisson_reconstruction
        faces, vertices = poisson_reconstruction(mesh_mid.vertices, mesh_mid.vertex_normals, depth=8)
        verts = torch.tensor(vertices, dtype=torch.float32, device='cuda')
        faces = torch.tensor(faces, dtype=torch.long, device='cuda')

        imesh = mesh.Mesh(verts, faces, material=material)
        # imesh = mesh.auto_normals(imesh)
        return imesh

    def render(self, glctx, target, lgt, opt_material, bsdf=None, if_normal=False, mode = 'geometry_modeling', if_flip_the_normal = False, if_use_bump = False):
        opt_mesh = self.getMesh(opt_material) 
        return render.render_mesh(glctx, 
                                  opt_mesh, 
                                  target['mvp'], 
                                  target['campos'], 
                                  lgt, 
                                  target['resolution'], 
                                  spp=target['spp'], 
                                  msaa= True,
                                  background= target['background'],
                                  bsdf= bsdf,
                                  if_normal= if_normal,
                                  # normal_rotate= target['normal_rotate'],
                                  normal_rotate=None,
                                  mode = mode,
                                  if_flip_the_normal = if_flip_the_normal,
                                  if_use_bump = if_use_bump
                                  )

        
    def tick(self, glctx, target, lgt, opt_material, iteration, if_normal, guidance, mode, if_flip_the_normal, if_use_bump):

        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        buffers = self.render(glctx, target, lgt, opt_material, if_normal= if_normal)

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        t_iter = iteration / self.FLAGS.iter

        # Image-space loss, split into a coverage component and a color component
        color_ref = target['img']
        norm_ref = target['norm']


        normal_pred = buffers['shaded'][..., :3] # -1~1
        bs, h, w, c = normal_pred.shape

        # conver normal_pred to canonical space
        front_mv = target['front_mv'][0, :3, :3] # [3, 3]
        rot = target['mv'][:, :3, :3] # bs, 3, 3
        rot = torch.matmul(torch.inverse(front_mv)[None, :, :], rot) # bs, 3, 3
        rot = rot.unsqueeze(1).repeat(1, h*w, 1, 1) # bs, h*w, 3, 3
        normal_pred = normal_pred.reshape(bs, -1, 3) # bs, h*w, 3
        normal_pred = torch.matmul(rot, normal_pred[:, :, :, None]) # bs, h*w, 3, 1
        normal_pred = (normal_pred.reshape(bs, h, w, c) + 1)*0.5 # bs, h, w, 3

        # if iteration % 100 == 0:
        #     save_image('tmp' + '/' + ('%04d_norm.png' % (iteration)), norm_ref[..., :3][0].detach().cpu().numpy())
        #     save_image('tmp' + '/' + ('%04d_norm_pred.png' % (iteration)), normal_pred[..., :3][0].detach().cpu().numpy())

        img_loss = torch.nn.functional.mse_loss(normal_pred, norm_ref[..., :3])

        # SDF regularizer
        reg_loss = None
        use_sdf_reg = False
        if use_sdf_reg:
            sdf_weight = self.FLAGS.sdf_regularizer - (self.FLAGS.sdf_regularizer - 0.01) * min(1.0, 4.0 * t_iter)
            reg_loss = sdf_reg_loss(self.sdf, self.all_edges).mean() * sdf_weight  # Dropoff to 0.01
            sdf_gradient_reg_loss = ((self.get_sdf_gradient().norm(dim=-1) - 1) ** 2).mean()
            reg_loss += sdf_gradient_reg_loss * 0.001

        # print("sdf_gradient_reg_loss:", sdf_gradient_reg_loss)

        if reg_loss is None:
            reg_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        sds_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        return sds_loss, img_loss, reg_loss



def save_image(fn, x):
    import imageio
    x = np.rint(x * 255.0)
    x = np.clip(x, 0, 255).astype(np.uint8)
    imageio.imsave(fn, x)