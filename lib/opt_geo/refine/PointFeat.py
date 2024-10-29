# Copyright (c) Alibaba, Inc. and its affiliates.

from pytorch3d.structures import Meshes, Pointclouds
import torch
# from lib.dataset.mesh_util import SMPLX, barycentric_coordinates_of_projection
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from pytorch3d import _C

_DEFAULT_MIN_TRIANGLE_AREA: float = 5e-3

def face_vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) *
                     nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, vertices.shape[-1]))

    return vertices[faces.long()]


# PointFaceDistance
class _PointFaceDistance(Function):
    """
    Torch autograd Function wrapper PointFaceDistance Cuda implementation
    """

    @staticmethod
    def forward(
        ctx,
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_points,
        min_triangle_area=_DEFAULT_MIN_TRIANGLE_AREA,
    ):
        """
        Args:
            ctx: Context object used to calculate gradients.
            points: FloatTensor of shape `(P, 3)`
            points_first_idx: LongTensor of shape `(N,)` indicating the first point
                index in each example in the batch
            tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
                triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
            tris_first_idx: LongTensor of shape `(N,)` indicating the first face
                index in each example in the batch
            max_points: Scalar equal to maximum number of points in the batch
            min_triangle_area: (float, defaulted) Triangles of area less than this
                will be treated as points/lines.
        Returns:
            dists: FloatTensor of shape `(P,)`, where `dists[p]` is the squared
                euclidean distance of `p`-th point to the closest triangular face
                in the corresponding example in the batch
            idxs: LongTensor of shape `(P,)` indicating the closest triangular face
                in the corresponding example in the batch.

            `dists[p]` is
            `d(points[p], tris[idxs[p], 0], tris[idxs[p], 1], tris[idxs[p], 2])`
            where `d(u, v0, v1, v2)` is the distance of point `u` from the triangular
            face `(v0, v1, v2)`

        """
        dists, idxs = _C.point_face_dist_forward(
            points,
            points_first_idx,
            tris,
            tris_first_idx,
            max_points,
            min_triangle_area,
        )
        ctx.save_for_backward(points, tris, idxs)
        ctx.min_triangle_area = min_triangle_area
        return dists, idxs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        min_triangle_area = ctx.min_triangle_area
        grad_points, grad_tris = _C.point_face_dist_backward(points, tris, idxs, grad_dists, min_triangle_area)
        return grad_points, None, grad_tris, None, None, None


def point_mesh_distance(meshes, pcls, weighted=True):

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()

    # point to face distance: shape (P,)
    point_to_face, idxs = _PointFaceDistance.apply(points, points_first_idx, tris, tris_first_idx, max_points, 5e-3)

    if weighted:
        # weight each example by the inverse of number of points in the example
        point_to_cloud_idx = pcls.packed_to_cloud_idx()  # (sum(P_i),)
        num_points_per_cloud = pcls.num_points_per_cloud()  # (N,)
        weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
        weights_p = 1.0 / weights_p.float()
        point_to_face = torch.sqrt(point_to_face) * weights_p

    return point_to_face, idxs


def barycentric_coordinates_of_projection(points, vertices):
    """https://github.com/MPI-IS/mesh/blob/master/mesh/geometry/barycentric_coordinates_of_projection.py"""
    """Given a point, gives projected coords of that point to a triangle
    in barycentric coordinates.
    See
        **Heidrich**, Computing the Barycentric Coordinates of a Projected Point, JGT 05
        at http://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf

    :param p: point to project. [B, 3]
    :param v0: first vertex of triangles. [B, 3]
    :returns: barycentric coordinates of ``p``'s projection in triangle defined by ``q``, ``u``, ``v``
            vectorized so ``p``, ``q``, ``u``, ``v`` can all be ``3xN``
    """
    # (p, q, u, v)
    v0, v1, v2 = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    u = v1 - v0
    v = v2 - v0
    n = torch.cross(u, v)
    sb = torch.sum(n * n, dim=1)
    # If the triangle edges are collinear, cross-product is zero,
    # which makes "s" 0, which gives us divide by zero. So we
    # make the arbitrary choice to set s to epsv (=numpy.spacing(1)),
    # the closest thing to zero
    sb[sb == 0] = 1e-6
    oneOver4ASquared = 1.0 / sb
    w = points - v0
    b2 = torch.sum(torch.cross(u, w) * n, dim=1) * oneOver4ASquared
    b1 = torch.sum(torch.cross(w, v) * n, dim=1) * oneOver4ASquared
    weights = torch.stack((1 - b1 - b2, b1, b2), dim=-1)
    # check barycenric weights
    # p_n = v0*weights[:,0:1] + v1*weights[:,1:2] + v2*weights[:,2:3]
    return weights


class PointFeat:

    def __init__(self, verts, faces):

        # verts [B, N_vert, 3]
        # faces [B, N_face, 3]
        # triangles [B, N_face, 3, 3]

        self.Bsize = verts.shape[0]
        self.device = verts.device
        self.faces = faces

        # SMPL has watertight mesh, but SMPL-X has two eyeballs and open mouth
        # 1. remove eye_ball faces from SMPL-X: 9928-9383, 10474-9929
        # 2. fill mouth holes with 30 more faces

        # if verts.shape[1] == 10475:
        #     faces = faces[:, ~SMPLX().smplx_eyeball_fid_mask]
        #     mouth_faces = (torch.as_tensor(SMPLX().smplx_mouth_fid).unsqueeze(0).repeat(self.Bsize, 1, 1).to(self.device))
        #     self.faces = torch.cat([faces, mouth_faces], dim=1).long()

        self.verts = verts.float()
        self.triangles = face_vertices(self.verts, self.faces)
        self.mesh = Meshes(self.verts, self.faces).to(self.device)

    def query(self, points):

        points = points.float()
        residues, pts_ind = point_mesh_distance(self.mesh, Pointclouds(points), weighted=False)

        closest_triangles = torch.gather(self.triangles, 1, pts_ind[None, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
        bary_weights = barycentric_coordinates_of_projection(points.view(-1, 3), closest_triangles)

        feat_normals = face_vertices(self.mesh.verts_normals_padded(), self.faces)
        closest_normals = torch.gather(feat_normals, 1, pts_ind[None, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
        shoot_verts = ((closest_triangles * bary_weights[:, :, None]).sum(1).unsqueeze(0))

        pts2shoot_normals = points - shoot_verts
        pts2shoot_normals = pts2shoot_normals / torch.norm(pts2shoot_normals, dim=-1, keepdim=True)

        shoot_normals = ((closest_normals * bary_weights[:, :, None]).sum(1).unsqueeze(0))
        shoot_normals = shoot_normals / torch.norm(shoot_normals, dim=-1, keepdim=True)
        angles = (pts2shoot_normals * shoot_normals).sum(dim=-1).abs()

        return (torch.sqrt(residues).unsqueeze(0), angles)
