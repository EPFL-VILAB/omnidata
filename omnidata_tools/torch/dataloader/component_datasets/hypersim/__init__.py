import functools
import json
import logging
import multiprocessing as mp
import os
from   pandas import read_csv
from   PIL import Image
from   pytorch3d.structures import Meshes
import torch
from   torchvision import transforms
import torchvision.transforms as T
from   typing      import Optional, List, Callable, Union, Dict, Any
import tqdm
from   tqdm.contrib.concurrent import process_map  # or thread_map
import h5py

from ...segment_instance import random_colors
from ...splits import get_splits
from ...omnidata_dataset import OmnidataDataset, load_subfolder
from ...transforms import transform_dense_labels, get_transform
###############################################################################
# Split info:
# Exports:
#   e.g. flat_split_to_spaces['tiny-train'] -> List[str] building names
###############################################################################
split_file = os.path.join(os.path.dirname(__file__), 'train_val_test_hypersim.csv')
__filedir__ = os.path.dirname(__file__)
metadata_camera_parameters_csv_file = os.path.join(os.path.dirname(__file__), 'metadata_camera_parameters.csv')
flat_split_to_spaces = get_splits(split_file)
buildings_to_exclude = []

###############################################################################
#  Semantic segmentation:
###############################################################################

CLASS_LABELS = [
    'undefined', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
    'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow',
    'mirror', 'floor-mat', 'clothes', 'ceiling', 'books', 'fridge', 'TV', 'paper', 'towel', 
    'shower-curtain', 'box', 'white-board', 'person', 'night-stand', 'toilet', 'sink', 'lamp',
    'bathtub', 'bag', 'other-struct', 'other-furntr', 'other-prop'
]

CLASS_LABEL_TRANSFORM = [
    0, 116, 87, 62, 41, 38, 39, 42, 85, 119, 122, 98, 123, 68, 82, 102, 78, 124, 99, 125, 92, 74, 
    79, 55, 54, 44, 96, 112, 126, 69, 127, 128, 94, 43, 53, 90, 64, 8, 0, 0, 0
]

NYU40_COLORS = [
    [ 0,    0,   0], [174, 199, 232], [152, 223, 138], [ 31, 119, 180], [255, 187, 120], [188, 189,  34],
    [140,  86,  75], [255, 152, 150], [214,  39,  40], [197, 176, 213], [148, 103, 189], [196, 156, 148],
    [ 23, 190, 207], [178,  76,  76], [247, 182, 210], [ 66, 188, 102], [219, 219, 141], [140,  57, 197],
    [202, 185,  52], [ 51, 176, 203], [200,  54, 131], [ 92, 193,  61], [ 78,  71, 183], [172, 114,  82],
    [255, 127,  14], [ 91, 163, 138], [153,  98, 156], [140, 153, 101], [158, 218, 229], [100, 125, 154],
    [178, 127, 135], [120, 185, 128], [146, 111, 194], [ 44, 160,  44], [112, 128, 144], [ 96, 207, 209],
    [227, 119, 194], [213,  92, 176], [ 94, 106, 211], [ 82,  84, 163], [100,  85, 144]
]

CLASS_COLORS = random_colors(len(NYU40_COLORS), bright=True, seed=99)


#############################
#      Hypersim 3D
#############################
DEFAULT_3D_DTYPE = torch.float64

# This coordinate transform is needed in order to convert the camera normals to world normals properly
coord_transform = torch.tensor([
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
], dtype=DEFAULT_3D_DTYPE)
# coord_transform = torch.tensor([
#     [-1.0, 0.0, 0.0],
#     [0.0, 0.0, -1.0],
#     [0.0, -1.0, 0.0],
# ], dtype=DEFAULT_3D_DTYPE)
# coord_transform          = torch.eye(3, dtype=DEFAULT_3D_DTYPE)
# coord_transform = torch.tensor([
#     [-1.0, 0.0, 0.0],
#     [0.0, 0.0, 1.0],
#     [0.0, 1.0, 0.0],
# ], dtype=DEFAULT_3D_DTYPE)
# torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=DEFAULT_3D_DTYPE)
coord_transform_k         = torch.eye(4, dtype=DEFAULT_3D_DTYPE)
coord_transform_k[:3, :3] = coord_transform
camera_convention_transform = torch.tensor([
    [-1.0,   0.0,    0.0,   0.0],
    [0.0,   1.0,    0.0,   0.0],
    [0.0,   0.0,    1.0,   0.0], 
    [0.0,   0.0,    0.0,   1.0]
], dtype=DEFAULT_3D_DTYPE)


# If doing center crop, need to adjust K, K_inv
def XA_b(A, B):
    x = torch.linalg.lstsq(A, B).solution
    res = (A @ x - B)
    return x.T, res

x0, x1, y0, y1, = 0.75, -0.75, 1.0, -1.0, 
A               = torch.tensor([
                    [x0, y0, 1],
                    [x0, y1, 1],
                    [x1, y0, 1],
                    [x1, y1, 1],
], dtype=DEFAULT_3D_DTYPE)
B               =  torch.tensor([
                    [ 1.,  1., 1],
                    [ 1., -1., 1],
                    [-1.,  1., 1],
                    [-1., -1., 1],
], dtype=DEFAULT_3D_DTYPE)
crop_inv_ndc    = XA_b(B, A)[0].T
crop_ndc        = torch.eye(4, dtype=DEFAULT_3D_DTYPE) 
crop_ndc[:3,:3] = crop_inv_ndc.inverse()
###############################################################################
# Make dataset 
###############################################################################
def semseg_remap_inplace(res):
    res2 = res.clone() 
    labels = torch.unique(res)
    for old_label in labels:
        if old_label == -1 or old_label == 255: continue
        res[res2 == old_label] = CLASS_LABEL_TRANSFORM[old_label]
    return res

@functools.lru_cache(None)
def get_camera_info(scene, camera):
    camera_positions_hdf5_file    = os.path.join(__filedir__, 'camera_keyframe', f'{scene}-{camera}', f'camera_keyframe_positions.hdf5')
    camera_orientations_hdf5_file = os.path.join(__filedir__, 'camera_keyframe', f'{scene}-{camera}', f'camera_keyframe_orientations.hdf5')
    with h5py.File(camera_positions_hdf5_file,    "r") as f: camera_positions    = f["dataset"][:]
    with h5py.File(camera_orientations_hdf5_file, "r") as f: camera_orientations = f["dataset"][:]
    
    # T = torch.tensor(camera_position_world).unsqueeze(0).float()
    # R = torch.tensor(R_world_from_cam).unsqueeze(0).float()
    R = torch.tensor(camera_orientations, dtype=DEFAULT_3D_DTYPE)
    # T = torch.tensor(camera_positions, dtype=DEFAULT_3D_DTYPE)
    T = -R.transpose(1,2).bmm(torch.tensor(camera_positions, dtype=DEFAULT_3D_DTYPE).unsqueeze(-1)).squeeze(-1)
    return T, R

@functools.lru_cache(None)
def get_metadata(scene_name):
    # read parameters from csv file
    df_camera_parameters = read_csv(os.path.join(__filedir__, 'metadata_camera_parameters.csv'), index_col="scene_name")
    df_ = df_camera_parameters.loc[scene_name]

    width_pixels          = int(df_["settings_output_img_width"])
    height_pixels         = int(df_["settings_output_img_height"])
    meters_per_asset_unit = df_["settings_units_info_meters_scale"]

    M_cam_from_uv = torch.tensor([[ df_["M_cam_from_uv_00"], df_["M_cam_from_uv_01"], df_["M_cam_from_uv_02"] ],
                            [ df_["M_cam_from_uv_10"], df_["M_cam_from_uv_11"], df_["M_cam_from_uv_12"] ],
                            [ df_["M_cam_from_uv_20"], df_["M_cam_from_uv_21"], df_["M_cam_from_uv_22"] ]], dtype=DEFAULT_3D_DTYPE)

    M_proj = torch.tensor([[ df_["M_proj_00"], df_["M_proj_01"], df_["M_proj_02"], df_["M_proj_03"] ],
                 [ df_["M_proj_10"], df_["M_proj_11"], df_["M_proj_12"], df_["M_proj_13"] ],
                 [ df_["M_proj_20"], df_["M_proj_21"], df_["M_proj_22"], df_["M_proj_23"] ],
                 [ df_["M_proj_30"], df_["M_proj_31"], df_["M_proj_32"], df_["M_proj_33"] ]], dtype=DEFAULT_3D_DTYPE)
    return dict( 
        width_pixels=width_pixels,
        height_pixels=height_pixels,
        M_cam_from_uv=M_cam_from_uv,
        M_proj=M_proj,
        meters_per_asset_unit=meters_per_asset_unit
    )

class HypersimDataset(OmnidataDataset):

    def _folder_in_split(self, folder, split):
        # return ('ai_037_002' in folder)
        # if 
        row = self.split_df.loc[ self.split_df['id']==folder.split('-')[0]]
        return (not row.empty and row.iloc[0][split] == 1)

    def _build_mesh_path(self, building): return  os.path.join(self.data_path, 'mesh', self.dataset_name, f'{building.split("-")[0]}.ply')


    def get_building_from_bpv(self, bpv):
        return bpv[0].split('-')[0]

    # def _point_info_supplement(self, point_info):
    #     point_info = super()._point_info_supplement(point_info)
    #     scene_name, cam_name = point_info['building'].split('-')
    #     point_info.update(get_metadata(scene_name))
    #     positions, orientations = get_camera_info(scene_name, cam_name)
    #     view = self._parse_fpath_for_view(point_info['path'])
    #     point_info['rotation_mat'] = orientations[int(view.point)]
    #     point_info['rotation_to_normal_mat'] = self.rotation_to_normal_mat
    #     return point_info
    
    def __init__(self, options: OmnidataDataset.Options, logger: logging.Logger=None):
        # if options.image_size is None:
        #     options.image_size = 512
        self.dataset_name = 'hypersim'
        self.rotation_to_normal_mat = torch.tensor([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float64)

        super().__init__(options, logger)
        if (self.num_positive > 1 or self.multiview_sampler is not None): 
            assert self.multiview_sampling_method in ['SHARED_PIXELS', 'CENTER_VISIBLE'], f'Hypersim multiview requires multiview_sampling_method is ["SHARED_PIXELS", "CENTER_VISIBLE"]--got {self.multiview_sampling_method}'
            if self.multiview_sampling_method == "CENTER_VISIBLE":
                assert self.sampled_camera_type in ['DIFFERENT', 'ANY', 'BACKOFF', 'SAME'], f'Hypersim sampled_camera_type is {self.sampled_camera_type}'
        if 'fragments' in self.transform: self.transform['fragments'] = get_transform('fragments', self.image_size, move_last_row=False)
        for task, _transform in self.transform.items():
            # if self.image_size is not None and task not in ['point_info', 'fragments']:
            if self.image_size is not None:
                _new_transform = []
                # if task == 'fragments': _new_transform.append(transforms.ToTensor())
                if task not in ['point_info']:
                    resize_method = T.InterpolationMode.BILINEAR if task in ['rgb'] else T.InterpolationMode.NEAREST
                    _new_transform = _new_transform + [
                        # transforms.Resize(self.image_size, resize_method), 
                        _transform,
                        transforms.CenterCrop(self.image_size),
                    ]
                # if task == 'semantic': _new_transform.append(semseg_remap_inplace)
                self.transform[task] = transforms.Compose(_new_transform)
                
    def _get_cam_to_world_R_T_K(self, point_info: Dict[str, Any], building: str, point: int, view: int, device='cpu') -> List[torch.Tensor]:
        scene, camera = building.split('-')
        T, R        = get_camera_info(scene, camera)
        metadata    = get_metadata(scene)
        K, scaling  = metadata['M_proj'], metadata['meters_per_asset_unit']
        K_inv       = metadata['M_cam_from_uv']
        T, R = T[int(point)] * scaling, R[int(point)]

        # Convert for mesh
        R           = coord_transform.unsqueeze(0) @ R @ coord_transform.T.unsqueeze(0)
        T           = (coord_transform.unsqueeze(0) @ T).squeeze()
        aspect_transform           = torch.eye(4, dtype=DEFAULT_3D_DTYPE)
        aspect_transform[0,0]      = 4.0 / 3.0 
        # aspect_transform[1,1]      = 0.5 #3.0 / 4.0 #K[0,1,1] / K[0,0,0]
        # aspect_transform[0,0]      = 4.0 / 3.0 #K[0,1,1] / K[0,0,0]
        
        # Adjust K with coordinates, croppping
        K     = crop_ndc @ camera_convention_transform @ aspect_transform @ K @ coord_transform_k.T
        K_inv = coord_transform @  K_inv @ camera_convention_transform[:3,:3].T @ crop_inv_ndc
        # K           = camera_convention_transform @ K @ coord_transform_k.T
        # K_inv       = coord_transform @ K_inv @ camera_convention_transform[:3,:3].T
        # T_inv       = -R.bmm(T.unsqueeze(0).unsqueeze(-1)).squeeze()
        # T_inv = torch.zeros_like(T_inv)
        return dict(cam_to_world_R=R.float().squeeze(), cam_to_world_T=T.float(), proj_K=K.float().squeeze(), proj_K_inv=K_inv.float().squeeze())

    def _load_mesh_postprocessing(self, mesh):
        verts = mesh.verts_list()[0]
        verts = torch.tensor(verts) @ coord_transform.T
        # x, y, z = verts.unbind(1)
        # verts = torch.stack((-x, z, y), 1)
        mesh = Meshes(verts=[verts], faces=[mesh.faces_list()[0]], textures=mesh.textures)
        return mesh
      