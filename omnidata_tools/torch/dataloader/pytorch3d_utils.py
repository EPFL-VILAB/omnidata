from   typing import (Optional, Union, List, Tuple)
from   pytorch3d.renderer.cameras import FoVPerspectiveCameras, CamerasBase
from   pytorch3d.common.datatypes import Device
from pytorch3d.renderer import (
    ray_bundle_to_ray_points,
    RayBundle,
)
from   pytorch3d.transforms import Rotate, Transform3d, Translate
import torch

class GenericPinholeCamera(CamerasBase):
    def __init__(self,
        R: torch.Tensor,
        T: torch.Tensor,
        K: torch.Tensor,
        K_inv: torch.Tensor=None,
        aspect_ratio = 1.0,
        device: Union[Device, str] = "cpu",
    ):
        """
            Args:
                R: Rotation matrix of shape (N, 3, 3)
                T: Translation matrix of shape (N, 3)
                K: A calibration matrix of shape (N, 4, 4)
                K_inv: (optional) A calibration matrix of shape (N, 3, 3)
                    If provided, this is used for unprojecting points
                aspect_ratio: 
                device: Device (as str or torch.device)
            """
        super().__init__(R=R, T=T, K=K, device=device)
        if K_inv is not None: assert K_inv.shape[-2:] == (3,3), f'{K_inv.shape}'
        self.K_inv = K_inv
  
  
    def get_ndc_to_view_direction_matrix(self, **kwargs) -> torch.Tensor:
        K_inv: torch.Tensor = kwargs.get("K_inv", self.K_inv)
        if K_inv is None: K_inv = self.get_projection_transform().get_matrix()[:,:3,:3].inverse()
        return K_inv


    def transform_ndc_to_view_direction(self, points_ndc, **kwargs) -> torch.Tensor:
        K_inv         = self.get_ndc_to_view_direction_matrix(**kwargs)
        batch_size    = K_inv.shape[0]
        points_view   = K_inv.bmm(points_ndc.reshape(batch_size, -1, 3).transpose(1,2)).transpose(1,2)
        points_view   = points_view / points_view.norm(dim=-1, keepdim=True)
        return points_view


    def camera_rays(self,
        height: int,
        width: int,
        world_coordinates: bool = True,
        device: Optional[Union[Device, str]] = None,
        **kwargs
    ) -> torch.Tensor:
        '''
            Args:
              world_coordinates: if False, in view space. Otherwise, rotated and translated by camera R,T
            Returns:
              camera_rays: camera rays coresponding to a grid of pixels [height,width]
        '''
        K_inv         = self.get_ndc_to_view_direction_matrix(**kwargs)
        batch_size    = K_inv.shape[0]
        device        = device if device is not None else self.device
        points_ndc    = create_grid_ndc(height, width, flatten=True, stacked=True, device=device, dtype=K_inv.dtype).unsqueeze(0)
        points_view   = self.transform_ndc_to_view_direction(points_ndc.expand(batch_size, -1, 3), **kwargs)
        if not world_coordinates: return points_view.reshape((batch_size, height, width, 3))
        world_to_view_transform = self.get_world_to_view_transform()
        return world_to_view_transform.inverse().transform_points(points_view).reshape((batch_size, height, width, 3))

    def camera_raybundle_world(self,
        xy_ndc:  torch.Tensor,
        lengths: torch.Tensor,
        device: Optional[Union[Device, str]] = None,
        **kwargs
    ) -> RayBundle:
        '''
            Args:
              xy_ndc: A tensor of shape [batch_size, n_points, 2|3] in NDC space. 
              lengs:  Lengths of each ray in raybundle
            Returns: 
              RayBundle corresponding to pixels reprojected into world space, according to lengths
        '''
        assert xy_ndc.ndim == 3, f"{xy_ndc.shape} has more than 3 dims! Expected [batch_size, n_points, 2|3]"
        if   xy_ndc.shape[-1] == 2: xy_ndc_hom = torch.cat([xy_ndc, torch.ones_like(xy_ndc[...,-1].unsqueeze(-1))], dim=-1)
        elif xy_ndc.shape[-1] == 3: xy_ndc_hom, xy_ndc = xy_ndc, xy_ndc[...,:2]
        else:                       raise ArgumentError("xy_ndc has unknown shape: {xy_ndc.shape} -- expected last dim to be 2 or 3")
        directions   = self.transform_ndc_to_view_direction(xy_ndc_hom, **kwargs)
        directions   = directions.bmm(self.R.transpose(1,2)) # for some reason, using self.get_world_to_view_transform(R=self.R, T=torch.zeros_like(self.T)) screws things up
        origins      = self.get_camera_center().unsqueeze(1).expand_as(directions)
        return RayBundle(
          origins    = origins,
          directions = directions, 
          lengths    = lengths, 
          xys        = xy_ndc,
        )


    def camera_raybundle_world_from_depth_euclidean(self,
        depth_euclidean: torch.Tensor,
        device: Optional[Union[Device, str]] = None,
        **kwargs,
    ) -> RayBundle:
        if depth_euclidean.ndim == 2: depth_euclidean = depth_euclidean.unsqueeze(0)
        assert depth_euclidean.ndim == 3, f'{depth_euclidean.ndim} != 3'
        batch_size, height, width = depth_euclidean.shape
        device, dtype             = depth_euclidean.device, depth_euclidean.dtype
        points_ndc                = create_grid_ndc(height, width, flatten=True, stacked=True, device=device, dtype=depth_euclidean.dtype)
        points_ndc                = points_ndc.unsqueeze(0).expand(batch_size, -1, 3)
        return self.camera_raybundle_world(points_ndc, depth_euclidean.reshape(batch_size, -1, 1), device=device, **kwargs)


    def unproject_points(self,
        xy_depth: torch.Tensor,
        world_coordinates: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("Cowardly refusing to unproject points not explicitly marked as Euclidean instead of Z-Buffer. Use unproject_metric_depth_euclidean() instead.")


    def unproject_metric_depth_euclidean(self,
        depth_euclidean: torch.Tensor,
        # fov_scaling: Union[Tuple, torch.Tensor]=((1.0,1.0),),
        # fov_shift: Union[Tuple, torch.Tensor]=((0.0,0.0),),
        world_coordinates: bool = True,
        **kwargs
    ) -> torch.Tensor:
        '''
            Args: 
              depth_euclidean: [batch_size, height, width]
              world_coordincates: 
            Returns:
              points_world: reprojected points in world|view space
        '''
        if depth_euclidean.ndim == 2: depth_euclidean = depth_euclidean.unsqueeze(0)
        assert depth_euclidean.ndim == 3, f'{depth_euclidean.ndim} != 3'
        batch_size, height, width = depth_euclidean.shape
        device, dtype             = depth_euclidean.device, depth_euclidean.dtype
        
        # If we knew that we wanted things always in world space, we could just use:
        # raybundle                 = self.camera_raybundle_world_from_depth_euclidean(depth_euclidean, device=device, **kwargs)
        # return ray_bundle_to_ray_points(raybundle).squeeze(2).reshape((batch_size, height, width, 3))
        
        # But to be safe we will do this manually and use self.get_world_to_view_transform()
        camera_rays               = self.camera_rays(height, width, world_coordinates=False, device=device).reshape(batch_size, -1, 3)
        camera_rays               = camera_rays * depth_euclidean.reshape(batch_size, -1, 1)
        if not world_coordinates: return camera_rays.reshape((batch_size, height, width, 3))
        world_to_view_transform = self.get_world_to_view_transform()
        return world_to_view_transform.inverse().transform_points(camera_rays).reshape((batch_size, height, width, 3))
  
#     def unproject_metric_depth_euclidean(self,
#         depth_euclidean: torch.Tensor,
#         fov_scaling: Union[Tuple, torch.Tensor]=((1.0,1.0),),
#         fov_shift: Union[Tuple, torch.Tensor]=((0.0,0.0),),
#         world_coordinates: bool = True,
#         **kwargs
#     ) -> torch.Tensor:
#         '''
#         Projects rays in NDC space [-1,1]x[-1,1]
#             fov_scaling: Scale each dimension of NDC space (used for crops) 
#             fov_shift:   Shift each dimension of NDC space (used for crops)
#             rays = rays_normalized * fov_scaling + fov_shift
#         '''
#         if depth_euclidean.ndim == 2: depth_euclidean = depth_euclidean.unsqueeze(0)
#         assert depth_euclidean.ndim == 3, f'{depth_euclidean.ndim} != 3'
#         batch_size, height, width = depth_euclidean.shape
#         device, dtype             = depth_euclidean.device, depth_euclidean.dtype
#         fov_scaling   = torch.tensor(fov_scaling, device=device, dtype=dtype).expand((batch_size, 2))
#         fov_shift     = torch.tensor(fov_shift, device=device, dtype=dtype).expand((batch_size, 2))
#         xx, yy        = create_grid_ndc(height, width, device=depth_euclidean.device, dtype=depth_euclidean.dtype)
#         xx            = xx.reshape(batch_size, -1) * fov_scaling[:,0].unsqueeze(-1) + fov_shift[:,0].unsqueeze(-1)
#         yy            = yy.reshape(batch_size, -1) * fov_scaling[:,1].unsqueeze(-1) + fov_shift[:,1].unsqueeze(-1)
#         points_ndc    = torch.stack([xx, yy, depth_euclidean.reshape(batch_size, -1)], dim=-1)
#         return self.unproject_points(points_ndc, world_coordinates=world_coordinates)

#     def unproject_points(self,
#         xy_depth: torch.Tensor,
#         world_coordinates: bool = True,
#         **kwargs,
#     ) -> torch.Tensor:
#         '''
#             Args:
#                 xy_depth: Points in NDC space, except last dimension is euclidean depth
#         '''
#         K_inv: torch.Tensor = kwargs.get("K_inv", self.K_inv)
#         if K_inv is None: K_inv = self.get_projection_transform().get_matrix()[:,:3,:3].inverse()
#         assert  xy_depth.ndim == 2 or xy_depth.ndim == 3
#         assert  xy_depth.shape[-1] == 3, f'{xy_depth.shape}'
#         if      xy_depth.ndim == 2: xy_depth = xy_depth.unsqueeze(0)
#         assert  xy_depth.ndim == 3
#         batch_size  = xy_depth.shape[0]
#         distance    = xy_depth[...,2]
#         xy_depth    = xy_depth.clone()
#         xy_depth[...,2] = 1.0
#         points_view = K_inv.bmm(xy_depth.reshape(batch_size, -1, 3).transpose(1,2)).transpose(1,2)
#         points_view = points_view / points_view.norm(dim=-1, keepdim=True) * distance.reshape(batch_size,-1).unsqueeze(-1)

#         if not world_coordinates: return points_view.reshape(xy_depth.shape)
#         world_to_view_transform = self.get_world_to_view_transform()
#         return world_to_view_transform.inverse().transform_points(points_view).reshape(xy_depth.shape)

    def get_projection_transform(self, **kwargs) -> Transform3d:
        """
        Calculate the perspective projection matrix with a symmetric
        viewing frustrum. Use column major order.
        The viewing frustrum will be projected into ndc, s.t.
        (max_x, max_y) -> (+1, +1)
        (min_x, min_y) -> (-1, -1)

        Args:
            **kwargs: parameters for the projection can be passed in as keyword
                arguments to override the default values set in `__init__`.

        Return:
            a Transform3d object which represents a batch of projection
            matrices of shape (N, 4, 4)

        .. code-block:: python

            h1 = (max_y + min_y)/(max_y - min_y)
            w1 = (max_x + min_x)/(max_x - min_x)
            tanhalffov = tan((fov/2))
            s1 = 1/tanhalffov
            s2 = 1/(tanhalffov * (aspect_ratio))

            # To map z to the range [0, 1] use:
            f1 =  far / (far - near)
            f2 = -(far * near) / (far - near)

            # Projection matrix
            K = [
                    [s1,   0,   w1,   0],
                    [0,   s2,   h1,   0],
                    [0,    0,   f1,  f2],
                    [0,    0,    1,   0],
            ]
        """
        K = kwargs.get("K", self.K)
        if K.shape != (self._N, 4, 4):
            msg = "Expected K to have shape of (%r, 4, 4)"
            raise ValueError(msg % (self._N))

        # Transpose the projection matrix as PyTorch3D transforms use row vectors.
        transform = Transform3d(
            matrix=K.transpose(1, 2).contiguous(), device=self.device
        )
        return transform

def create_grid_ndc(height_pixels, width_pixels, stacked=False, flatten=False, **kwargs):
    '''
        Returns rays that cast pixels in image.
        Order is consisteny with omnidata dataloader
        Args:
            width_pixels: int
            height_pixels: int
            **kwargs: Device, dtype, etc.
    '''
    # create grid of uv-values
    u_min, u_max   = -1.0, 1.0
    v_min, v_max   = -1.0, 1.0
    half_du = 0.5 * (u_max - u_min) / width_pixels
    half_dv = 0.5 * (v_max - v_min) / height_pixels
    yy, xx  = torch.meshgrid(
                torch.linspace(v_max-half_dv, v_min+half_dv, height_pixels, **kwargs),
                torch.linspace(u_max-half_du, u_min+half_du, width_pixels, **kwargs),
                indexing='ij')
    if flatten: xx, yy = xx.flatten(), yy.flatten()
    if stacked: return torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    return xx, yy

#     def get_grid_ndc(self, height, width, device, dtype):
#         xx, yy        = create_grid_ndc(height, width, device=device, dtype=dtype)
#         xx            = xx.flatten()
#         yy            = yy.flatten()
#         points_ndc    = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
#         return points_ndc



##################################
# Visualization
##################################
import copy, math, torch
from   torch import Tensor
from   pytorch3d.structures import Meshes, Pointclouds
from   pytorch3d.structures.pointclouds import join_pointclouds_as_batch
from   pytorch3d.common.datatypes import Device
from   pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene

def get_batch_cam_params(batch):
    if 'positive' in batch: raise ValueError('batch has key "positive"--just pass in each components please')
    return_dict = {}
    n_views = len(batch['point_info'])
    for k in ['cam_to_world_R', 'cam_to_world_T', 'proj_K', 'proj_K_inv']:
        return_dict[k] = torch.stack([batch['point_info'][i][k] for i in range(n_views)], dim=1)
    return return_dict

def unproject_to_pointclouds(
    cam_to_world_R: Tensor,  # [V, 3, 3]
    cam_to_world_T: Tensor,  # [V, 3]
    proj_K:         Tensor,  # [V, 4, 4]
    proj_K_inv:     Tensor,  # [V, 3, 3]
    distance:       Tensor,  # [V, H, W]
    features:       Tensor,  # [V, C, H, W]
    mask_valid:     Optional[Tensor]=None,  # [V, H, W]
):
    assert cam_to_world_R.ndim == 3, f'{cam_to_world_R.shape}'
    assert cam_to_world_T.ndim == 2, f'{cam_to_world_T.shape}'
    assert proj_K.ndim == 3,         f'{proj_K.shape}'
    assert proj_K_inv.ndim == 3,     f'{proj_K_inv.shape}'
    assert distance.ndim == 3,   f'{distance.shape}'
    assert features.ndim == 4,   f'{features.shape}'
    assert mask_valid.ndim == 3, f'{mask_valid.shape}'
    features     = features.permute(0,2,3,1)
    cameras      = GenericPinholeCamera(R=cam_to_world_R, T=cam_to_world_T, K=proj_K, K_inv=proj_K_inv, device=distance.device)
    world_points = cameras.unproject_metric_depth_euclidean(distance, world_coordinates=True).reshape(len(cameras), -1, 3)
    if mask_valid is None: return Pointclouds(points=[world_points], features=[features])
    points = [feats[keep] for feats, keep in zip(features, mask_valid)]
    # for pts in points: print(pts.shape)
    return Pointclouds(
        points   = [pts[keep.reshape(-1)]   for pts, keep in zip(world_points, mask_valid)],
        features = [feats[keep] for feats, keep in zip(features, mask_valid)]
    )

def batch_unproject_to_multiview_pointclouds(
    cam_to_world_R: Tensor,
    cam_to_world_T: Tensor,
    proj_K:         Tensor,
    proj_K_inv:     Tensor,
    distance:       Tensor,
    features:       Tensor,
    mask_valid:     Optional[Tensor]=None,
):
    if mask_valid is None: mask_valid = [None] * len(cam_to_world_R)
    return [
        unproject_to_pointclouds(
            cam_to_world_R = _cam_to_world_R,
            cam_to_world_T = _cam_to_world_T,
            proj_K         = _proj_K,
            proj_K_inv     = _proj_K_inv,
            distance       = _distance,
            features       = _features,
            mask_valid     = _mask_valid,
        ) 
        for (_cam_to_world_R, _cam_to_world_T, _proj_K, _proj_K_inv, _distance, _features, _mask_valid) \
        in zip(cam_to_world_R, cam_to_world_T, proj_K, proj_K_inv, distance, features, mask_valid)
    ]

#export
def subsample_multivew_pcs_batch(pcs_batch, max_per_view=-1):
    '''
        Given a list of pointcloud lists (all of same length): [pc1], [pc2], ..., [pcN]
        Subsamples all pointclouds (matched by index)
    '''
    if max_per_view < 0: return pcs_batch
    if any(len(pcs_batch[0])!= len(i) for i in pcs_batch): raise ValueError(f"Trying to subsamples matched pointclouds with differing batch sizes: {[len(i) for i in pcs_batch]}")
    pcs_out = [subsample_multivew_pcs(pcs_all=pcs_example, max_per_view=max_per_view) for pcs_example in zip(*pcs_batch)]
    return zip(*pcs_out)

def subsample_multivew_pcs(pcs_all, max_per_view):
    '''
        Given a list of pointclouds pc1, pc2, ..., pcN
        Subsamples all pointclouds (matched by index)
    '''
    pc_examplar = pcs_all[0]
    indices = [torch.arange(l, device=pc_examplar.device).unsqueeze(-1) for l in pc_examplar.num_points_per_cloud()]
    pc_idx  = Pointclouds(points=pc_examplar.points_list(), features=indices).subsample(max_per_view)
    idxs_list = [idxs.squeeze(-1) for idxs in pc_idx.features_list()]
    return [Pointclouds(
            points=[p[idxs] for (p, idxs) in zip(pcs.points_list(), idxs_list)],
            features=[f[idxs] for (f, idxs) in zip(pcs.features_list(), idxs_list)])
        for pcs in pcs_all]


plotly_mesh_kwargs = dict(
    xaxis={"backgroundcolor":"rgb(200, 200, 230)"},
    yaxis={"backgroundcolor":"rgb(230, 200, 200)"},
    zaxis={"backgroundcolor":"rgb(200, 230, 200)"}, 
    axis_args=AxisArgs(showgrid=True)
)
plotly_pc_kwargs = dict(
    xaxis={"backgroundcolor":"rgb(200, 200, 230)"},
    yaxis={"backgroundcolor":"rgb(230, 200, 200)"},
    zaxis={"backgroundcolor":"rgb(200, 230, 200)"}, 
    axis_args=AxisArgs(showgrid=True),
    scaleratio = 1, point_size=30,
)


def show_batch_pc(batch, batch_idx, view_idxs, view_kwargs=plotly_pc_kwargs, figsize=750):
    if isinstance(view_idxs, int): view_idxs = [view_idxs]
    pos        = batch.get('positive', batch)
    bpv        = pos['building'][batch_idx], pos['point'][batch_idx], pos['view'][batch_idx]
    dataset    = pos['dataset'][batch_idx]
    mask_valid = pos['mask_valid'].bool()[batch_idx,view_idxs]#.squeeze(1)
    distance   = pos['depth_euclidean'][batch_idx,view_idxs]#.squeeze(1)
    rgb        = pos['rgb'][batch_idx,view_idxs].unsqueeze(1)
    cam_params = { k: v[batch_idx,view_idxs].unsqueeze(1)
                   for (k, v) in get_batch_cam_params(pos).items()}
    pcs_full   = batch_unproject_to_multiview_pointclouds(mask_valid=mask_valid, distance=distance, features=rgb, **cam_params)
    pcs_full   = join_pointclouds_as_batch(pcs_full)
    fig        = plot_scene({ f"{dataset}: {bpv}": { "Joined": pcs_full, },}, **view_kwargs)
    fig.update_layout(height=figsize, width=figsize)
    fig.show()

def show_batch_scene(batch, batch_idx, view_idxs, view_kwargs=plotly_pc_kwargs, figsize=750):
    if isinstance(view_idxs, int): view_idxs = [view_idxs]
    pos        = batch.get('positive', batch)
    bpv        = pos['building'][batch_idx], pos['point'][batch_idx], pos['view'][batch_idx]
    dataset    = pos['dataset'][batch_idx]
    mask_valid = pos['mask_valid'].bool()[batch_idx,view_idxs]#.squeeze(1)
    distance   = pos['depth_euclidean'][batch_idx,view_idxs]#.squeeze(1)
    rgb        = pos['rgb'][batch_idx,view_idxs].unsqueeze(1)
    cam_params = { k: v[batch_idx,view_idxs].unsqueeze(1)
                   for (k, v) in get_batch_cam_params(pos).items()}
    pcs_full   = batch_unproject_to_multiview_pointclouds(mask_valid=mask_valid, distance=distance, features=rgb, **cam_params)
    cameras    = GenericPinholeCamera(
                    R=cam_params['cam_to_world_R'].squeeze(1),
                    T=cam_params['cam_to_world_T'].squeeze(1),
                    K=cam_params['proj_K'].squeeze(1),
                    K_inv=cam_params['proj_K_inv'].squeeze(1),
                    device=distance.device)
    pcs_full   = join_pointclouds_as_batch(pcs_full)
    fig        = plot_scene({ f"{dataset}: {bpv}": { 
                                "Points": pcs_full, 
                                "cameras": cameras,
                            },},
                            **view_kwargs)
    fig.update_layout(height=figsize, width=figsize)
    fig.show()


def show_pc(pc=None, points=None, colors=None, view_kwargs=plotly_pc_kwargs, figsize=750):
    if pc is None: pc = Pointclouds(points=points, features=colors)
    fig        = plot_scene({ "Merged pointcloud": { "Joined": pc, },}, **view_kwargs)
    fig.update_layout(height=figsize, width=figsize)
    fig.show()
