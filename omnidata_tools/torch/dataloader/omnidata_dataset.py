'''
Base Omnidata dataset. 

'''
import copy
import functools as ft
import glob
import json
import loguru
import logging
import math
import multiprocessing as mp
import os
import pickle
import numpy as np
import pandas as pd
from   pytorch3d.structures import Meshes, Pointclouds
from   pytorch3d.transforms import euler_angles_to_matrix
from   pytorch3d.renderer import FoVPerspectiveCameras
import random

import re
import torch
import torch.utils.data as data
import torch.nn.functional as F
import hashlib
import warnings

from  collections import namedtuple, Counter, defaultdict
from  dataclasses import dataclass, field
# from  joblib      import Parallel, delayed
from  PIL         import ImageFile
from  time        import perf_counter 
from  tqdm        import tqdm
from  torchvision import transforms
from  typing      import Optional, List, Callable, Union, Dict, Any, Tuple
from  tqdm.contrib.concurrent import process_map, thread_map
from .transforms       import default_loader, get_transform
from .task_configs     import task_parameters, SINGLE_IMAGE_TASKS
from .segment_instance import COMBINED_CLASS_LABELS
from .scene_metadata import BuildingMetadata, BuildingMultiviewMetadata

ImageFile.LOAD_TRUNCATED_IMAGES = True # TODO Test this
LabelFile   = namedtuple('LabelFile', ['point', 'view', 'domain'])
View        = namedtuple('View', ['building', 'point', 'view'])
BPV         = namedtuple('BPV', ['building', 'point', 'view'])
MAX_VIEWS   = 45
N_OUTPUTS   = {
               'segment_semantic': len(COMBINED_CLASS_LABELS)-1,
               'depth_zbuffer': 1, 
               'normal': 3,
               'edge_occlusion': 1,
               'edge_texture': 1,
               'keypoints3d':1,
               'principal_curvature': 3
}


class OmnidataDataset(data.Dataset):
    @dataclass
    class Options():
        # Core options: where is the data located, and what do you want out of it.
        data_path:       str
        split:           str              = 'train'
        data_amount:     str              = 'tiny'
        tasks:           List[str]        = field(default_factory=lambda: ['rgb'])
        image_size:      Optional[int]    = None
        
        # Performance
        # Load times can be a few minutes, even when parallelized. This should help.
        n_workers:       Optional[int]    = None   # Used to build BPV index, multiview indices
        cache_dir:       Optional[str]    = None   # Save indices to disk. None to use default, False to disable caching
        overwrite_cache: bool             = False  # 

        # Multiview land
        num_positive:        Union[int, str] = 1    # Either int or 'all'. >1 enters multiview land
        multiview_sampling_method: str       = 'FILENAME' # None, FILENAME, SHARED_PIXELS, CENTER_VISIBLE
        min_views:           int             = 1           # Drop BPVs with too few multiview candidates. Ignored with CENTER_VISIBLE, which just throws errors. 
        max_views:           int             = MAX_VIEWS   # Drop excessive multiview candidates. Used by SHARED_PIXELS to avoid storing all (billions) of pairs.
        min_overlap_prop:    float           = 0.25        # Usable only with SHARED_PIXELS.
        multiview_path_len:  int             = 1           # Usable only with CENTER_VISIBLE.
        sampled_camera_type: str             = 'BACKOFF' # Usable only with CENTER_VISIBLE. Could be extended to SHARED_PIXELS.
        sampled_camera_knn:  Optional[int]   = -1        # Usable only with CENTER_VISIBLE. Could be extended to SHARED_PIXELS.
        backoff_order:       List[str]       = field(default_factory=lambda: ['SAME', 'FIXATED', 'DIFFERENT'])

        # Mesh land
        load_building_meshes:  bool          = False
        load_mesh_textures:    bool          = False
        mesh_cache_size:       Optional[int] = None        # Set to None to save everything. 0 to disable caching
        
        # Advanced options. (Sasha: i.e. I wrote this code and never used these options)
        transform:       Optional[Union[Dict[str, Callable], str]] = "DEFAULT"    # None, "DEFAULT", or {TASK_NAME: transform_callable}
        make_dataset_fn: Optional[Callable] = None        
        
        # deprecated
        normalize_rgb: bool = False


    def __init__(self, options: Options, logger=None):
        '''
            Pytorch base class for an Omnidata dataset
            The base class provides
              - __getitem__
              - Loading multiview
            This expects that the data is structured
                /path/to/data/
                    - Domain (Modality)
                       - Dataset
                           - Group/Building
                               - point_i_view_j_domain_DOMAIN.png (or equivalent extension)

            You can add other modalitites or cached encoding, then they can be added into the directory as:
                /path/to/data/
                    rgb_encoding/
                        modelk/
                            point_i_view_j.npy
                    ...

            Basically, any other folder name will work as long as it is named the same way.
        '''
        if isinstance(options.tasks, str): # interpret tasks="point_info" as  tasks=["point_info"]
            options.tasks = [options.tasks]
            options.transform = {options.tasks[0]: options.transform}        

        self.cache_dir            = options.cache_dir if options.cache_dir else os.path.join(options.data_path, '.data_cache')
        self.data_path            = options.data_path
        self.data_amount          = options.data_amount
        self.overwrite_cache    = options.overwrite_cache
        self.image_size           = options.image_size
        self.logger               = loguru.logger if logger is None else logger
        self.load_mesh_textures   = options.load_mesh_textures
        self.mesh_cache_size      = options.mesh_cache_size
        self.n_workers            = options.n_workers if options.n_workers is not None else mp.cpu_count()
        self.split                = options.split
        self.split_file           = os.path.join(os.path.dirname(__file__), 'component_datasets', self.dataset_name, f'train_val_test_{self.dataset_name}.csv')
        self.split_df             = pd.read_csv(self.split_file) 
        self.tasks                = sorted(options.tasks)
        self.transform            = options.transform
        
        # Multiview
        self.num_positive         = MAX_VIEWS if options.num_positive == 'all' else options.num_positive
        self.multiview_sampling_method = options.multiview_sampling_method
        self.sampled_camera_type  = options.sampled_camera_type
        self.min_views            = options.min_views
        self.max_views            = options.max_views
        self.min_overlap_prop     = options.min_overlap_prop
        self.multiview_path_len   = options.multiview_path_len
        self.sampled_camera_knn   = options.sampled_camera_knn
        self.backoff_order        = tuple(options.backoff_order)
        # self.cooccurrence_method  = options.cooccurrence_method
        # self.filter_missing_cooccurrence = options.filter_missing_cooccurrence
        # self.randomize_views      = options.randomize_views


        if self.num_positive > 1 and self.multiview_sampling_method is None:
            raise ArgumentError(f'If using multiview (num_positive={self.num_positive} > 1, multiview_sampling_method cannot be None!')
        
        if self.multiview_sampling_method is None:
            self.multiview_sampler = None
        else:
            self.multiview_sampler = MULTIVIEW_SAMPLING_METHODS[self.multiview_sampling_method](
                min_views=self.min_views,
                max_views=self.max_views,
                min_overlap_prop=self.min_overlap_prop,
                sampled_camera_type=self.sampled_camera_type, 
                path_length=self.multiview_path_len,
                backoff_order=self.backoff_order,
                sampled_camera_knn=self.sampled_camera_knn,
            )
        

        # CACHE
        self.cache_enabled = (self.cache_dir is not False)
        self.cache_path    = False if not self.cache_enabled else os.path.join(
            self.cache_dir,
            f'{type(self).__name__}_{self.data_amount}_{self.split}'
        )
        self.cache = FileDirCache(self.cache_path, self.overwrite_cache, cache_enabled=self.cache_enabled)
        ############# END REFACTOR


        # Execute init
        start_time = perf_counter()
        self.setup_transform()
        # self.make_bpv_tables()
        if self.bpv_tables_cache_key in self.cache:
            self.load_bpv_tables()
        else: 
            self.make_bpv_tables()
            self.save_bpv_tables()
        # self.load_mesh = ft.lru_cache(maxsize=self.mesh_cache_size)(self._load_mesh)
        self.load_meshes()
        self.validate()        
        self.num_points    = len(self.views)
        self.num_images    = len(self.bpv_list)
        self.num_buildings = len(self.bpv_dict)
        end_time = perf_counter()
        self.logger.success("Loaded {} images in {:0.2f} seconds".format(self.num_images, end_time - start_time))
        self.logger.success(f"\t ({self.num_buildings} buildings) ({self.num_points} points) ({self.num_images} images) for domains {self.tasks}")


    def __getitem__(self, index, other_bpvs: Optional[List]=None):
        building, point, view = self.bpv_list[index]
        positive_bpv = [(building, point, view)] # positive_bpv[0]: Anchor building / point / view
        positive_samples = {}
        # Just do uniform sampling for now. Chould reweight based on overlap.
        if other_bpvs is not None:
            positive_bpv = positive_bpv + other_bpvs
        elif self.num_positive > 1 and self.sampled_camera_type == 'SAME':
            positive_bpv = positive_bpv * self.num_positive
        elif self.num_positive > 1: # get other positive views
            other_bpvs = self.multiview_sampler.sample(positive_bpv[0], self.num_positive-1)
            positive_bpv = positive_bpv + other_bpvs

        for task in self.tasks:
            task_samples = []
            for b,p,v in positive_bpv:
                res = default_loader(self.url_dict[(task, b, p, v)])
                if self.transform is not None and self.transform[task] is not None: res = self.transform[task](res)
                if task == 'point_info':
                    # nfp = res.get('nonfixated_points_in_view', [])
                    res = self._get_cam_to_world_R_T_K(res, b, p, v) 
                    # res.update(self._get_world_to_cam_R_T_K(res, b, p, v))
                    # res.update(dict(building=b, point=p, view=v, nonfixated_points_in_view=nfp))
                    res.update(dict(building=b, point=p, view=v))
                task_samples.append(res)
            if task != 'point_info':
                task_samples = torch.stack(task_samples)
                # task_samples = torch.stack(task_samples) if self.num_positive > 1 else task_samples[0]
            positive_samples[task]   = task_samples
        positive_samples['point']    = point
        positive_samples['building'] = building
        positive_samples['view']     = view
        positive_samples['dataset']  = self.__class__.__name__
        return {'positive': positive_samples}

    def __len__(self): return len(self.bpv_list)


    #####################################
    #       BPV Indexing Structs        #
    #####################################    
    @property
    def bpv_tables_cache_key(self): 
        ''' Hashes all relevant settings'''
        mvsck = "no_multiview" if self.multiview_sampler is None else self.multiview_sampler.cache_key
        full_key = "_".join(self.tasks) + "__" + mvsck
        return 'settings_md5_' + hashlib.md5(full_key.encode('utf-8')).hexdigest()

    def save_bpv_tables(self):
        ''' Saves synced single- and multi-view info from disk.'''
        self.logger.info(f"Saving BPV tables to {self.cache.keypath(self.bpv_tables_cache_key)}...")
        multiview_sampler_data = self.multiview_sampler.state_dict() if self.multiview_sampler is not None else None
        self.cache.put(self.bpv_tables_cache_key, dict(
            bpv_list = self.bpv_list,
            urls     = self.urls,
            multiview_sampler_data = multiview_sampler_data
        ))
        self.logger.info(f"Done saving BPV tables to {self.cache.keypath(self.bpv_tables_cache_key)}...")

    def load_bpv_tables(self):
        '''
            Loads complete, synced single- and multi-view info from disk.
            Since the slow part seems to be creating the extended structs, we don't save much here.
        '''
        self.logger.info(f"Loading bpv tables from {self.cache.keypath(self.bpv_tables_cache_key)}...")
        results = self.cache[self.bpv_tables_cache_key]
        self.bpv_list = results['bpv_list']
        
        self.url_dict = {}
        self.urls = results['urls']
        for task in self.tasks:
            for url in self.urls[task]:
                bpv    = self.get_bpv(url)
                self.url_dict[(task, *bpv)] = url

        if self.multiview_sampler is not None:
            self.multiview_sampler.load_state_dict(results['multiview_sampler_data'])

        # Save building -> point -> view dict
        dd_list = ft.partial(defaultdict, list)
        self.bpv_dict, self.views = defaultdict(dd_list), dd_list()
        for building, point, view in self.bpv_list:
            self.views[(building, point)].append(view)
            self.bpv_dict[building][point].append(view)
        self.logger.info("Loaded bpv tables...")

    def make_bpv_tables(self):
        # Find all images in self.data_dir
        urls = {}
        for task in self.tasks:
            key = f'{task}_urls'
            if key in self.cache: 
                self.logger.info(f'Loading {key} from {self.cache.keypath(key)}.')
                urls[task] = self.cache[key]
            else:
                self.logger.info(f'Recomputing {key}.')
                urls[task] = self.cache.put(key, self.make_task_dataset(task=task))
        
        # Filter out images that are not present in all directories
        self.urls, dataset_size  = self._remove_unmatched_images(urls)

        # Saving some lists and dictionaries for fast lookup
        self.url_dict  = {}  # Save (task, building, point, view) -> URL dict
        bpv_count = {} # Dictionary to check if all (building, point, view) tuples have all tasks
        for task in self.tasks:
            for url in urls[task]:
                bpv    = self.get_bpv(url)
                self.url_dict[(task, *bpv)]  = url
                bpv_count[bpv]          = bpv_count.get(bpv, 0) + 1
        self.bpv_list = sorted([bpv_tuple for bpv_tuple, count in bpv_count.items() if count == len(self.tasks)])

        # If using multiview information, make sure the two are calibrated.
        if self.multiview_sampler is not None:
            key = self.multiview_sampler.cache_key
            if key in self.cache:
                self.logger.info(f'Loading multiview tables from {self.cache.keypath(key)}.')
                self.multiview_sampler.load_state_dict(self.cache[key])
            else:
                self.logger.info(f'Computing multiview tables (cache key: {key}) .')
                self.multiview_sampler.tables_create(self, self.n_workers)
                self.cache.put(key, self.multiview_sampler.state_dict())
            self.bpv_list = self.multiview_sampler.tables_sync(self, self.bpv_list, n_workers=self.n_workers)

        self.logger.info("Creating extended BPV structs...")
        # Save building -> point -> view dict
        dd_list = ft.partial(defaultdict, list)
        self.bpv_dict, self.views = defaultdict(dd_list), dd_list()
        for building, point, view in self.bpv_list:
            self.views[(building, point)].append(view)
            self.bpv_dict[building][point].append(view)
        self.logger.info("Done building BPV indexes...")

    def make_task_dataset(self, task):
        '''
            Args: 
                task: str (e.g. depth_euclidean). Usually the name of the subfolder where this task lives
            
            Returns:
                data_paths: List[str]. A list of paths to different data objects. 
                    These paths are matched across tasks via self.get_bpv
        '''
        dirpath         = os.path.expanduser(os.path.join(self.data_path, task, self.dataset_name))
        folders         = os.listdir(dirpath)
        subfolder_paths = [ os.path.join(dirpath, folder)  
            for folder in folders
            if os.path.isdir(os.path.join(dirpath, folder)) and self._folder_in_split(folder, self.split)
        ]
        if not os.path.isdir(dirpath): raise ValueError(f'Expected to find data directory in {dirpath}, but that is not a directory.')
        if self.n_workers == 1:
            images = [ glob.glob(os.path.join(subfolder, '*'))
                 for subfolder in tqdm(subfolder_paths, desc=f'Loading {task} paths')]
        else:
            images = thread_map(glob.glob, 
                [os.path.join(subfolder, '*') for subfolder in subfolder_paths],
                desc        = f'Loading {task} paths ({self.n_workers} workers)',
                max_workers = self.n_workers,
                chunksize   = min(1, len(subfolder_paths) // self.n_workers + 1))
        return list(sum(images, start=[]))

    def _remove_unmatched_images(self, dataset_urls) -> (Dict[str, List[str]], int):
        '''
            Filters out point/view/building triplets that are not present for all tasks
            
            Returns:
                filtered_urls: Filtered Dict
                max_length: max([len(urls) for _, urls in filtered_urls.items()])
        '''
        # Check if we even need to do anything
        n_images_task                 = [(len(obs), task) for task, obs in dataset_urls.items()]
        min_images, max_images        = min(n_images_task)[0], max(n_images_task)[0]
        if min_images == max_images:  return dataset_urls, max_images
        else: self.logger.error(f"Unequal # of images per modality [max difference {(100.0 * (max_images - min_images)) / max_images:0.2f} %]. Keeping intersection: \n\t" \
                                         + '\n\t '.join([f'{t}: {len(obs)}' for t, obs in dataset_urls.items()]))
        intersection = set.intersection(*[set(map(self.get_bpv, paths)) for paths in dataset_urls.values()])
        new_urls = {
            task: list(filter(lambda path: self.get_bpv(path) in intersection, paths))
            for task, paths in dataset_urls.items()
        }
        self.logger.error(f' _remove_unmatched_images: ({len(intersection)} images/task)...')
        return new_urls, len(intersection)

    ################## Load meshes
    def _load_mesh(self, building, overwrite_pkl=False):
        '''
            Returns a pytorch3d.structures.meshes.Meshes object for the building. 
        '''
        pkl_path = self._get_mesh_pkl_path(building)
        if os.path.exists(pkl_path) and not overwrite_pkl:
            with open(pkl_path, 'rb') as f:
                mesh = pickle.load(f)
        else:
            from pytorch3d.structures import Meshes
            # from pytorch3d.io import load_obj, load_ply
            filename = self._get_mesh_path(building)
            if self.load_mesh_textures:
                from pytorch3d.io import IO
                mesh = IO().load_mesh(filename)
            else:
                import trimesh
                mesh = trimesh.load(filename, process=False, maintain_order=True)
                mesh = Meshes(
                  verts=[torch.tensor(mesh.vertices, dtype=torch.float32)],
                  faces=[torch.tensor(mesh.faces, dtype=torch.long)])
            mesh = self._load_mesh_postprocessing(mesh)
            # In case we want to cache something, do that here.
            os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
            with open(pkl_path, 'wb') as f:
                pickle.dump(mesh, f)
        return mesh
    
    def load_meshes(self):
        return
        # Could set up meshes
        self.meshes = {}
        if self.options.load_building_meshes:
            self.logger.info("Loading meshes...", enqueue=True)
            building_mesh_pairs = Parallel(n_jobs=mp.cpu_count(), prefer='threads')(
                delayed(self.load_mesh)(building) for building in tqdm(self.bpv_dict)
            )
            self.meshes.update(dict(building_mesh_pairs))         
            self.logger.info("Loading meshes...", enqueue=True)

    def _load_mesh_postprocessing(self, mesh):
        verts = mesh.verts_list()[0]
        x, y, z = verts.unbind(1)
        verts = torch.stack((-x, z, y), 1)
        mesh = Meshes(verts=[verts], faces=[mesh.faces_list()[0]], textures=mesh.textures)
        return mesh

    def _get_mesh_pkl_path(self, building):
        return os.path.join(self.cache_dir, 'mesh_cache', self.dataset_name, f"{building}.pkl")
    
    def _get_mesh_path(self, building):
        return os.path.join(self.data_path, 'mesh', self.dataset_name, f"{building}.ply")
    ################## END load meshes
    
    def validate(self):
        self.logger.trace("Running post-setup validation.")
        # Freeze things
        for v in self.bpv_dict.values():
            v.default_factory = None
        self.views.default_factory = None

        failed = False
        # if self.num_positive > 1 or self.filter_missing_cooccurrence:
        #     assert len(self.bpv_cooccurrence) == len(self.bpv_list), f"{len(self.bpv_cooccurrence)} == {len(self.bpv_list)}"
        
        dataset_len = None
        for task, values in self.urls.items():
            dataset_len = len(values) if dataset_len is None else dataset_len
            if len(values) != dataset_len: 
                self.logger.error(f'self.urls[{task}] has {len(values)} items, but one other task has {dataset_len}')
                failed = True
        
        # for b, p, v in self.bpv_list:
        #     for task in self.tasks:
        #         assert (task, b, p, v) in self.url_dict, (task, b, p, v)
        self.url_dict = pd.Series(self.url_dict) # To prevent copy-on-access in __getitem__ when using mp.set_start_method('fork')
        self.bpv_list = pd.Series(self.bpv_list) # To prevent copy-on-access in __getitem__ when using mp.set_start_method('fork')
        if hasattr(self.multiview_sampler, 'scene_metadata'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                for v in self.multiview_sampler.scene_metadata.values(): v.freeze()
                for v in self.multiview_sampler.scene_mv_metadata.values(): v.freeze()

        self.logger.trace("Finished post-setup validation.")
        if failed: raise RuntimeError("Failed dataset validation")


    def randomize_order(self, seed=0):
        random.seed(seed)
        random.shuffle(self.bpv_list)
    

    def setup_transform(self):
        if isinstance(self.transform, str):
            if self.transform == 'DEFAULT': self.transform = {task: get_transform(task, self.image_size) for task in self.tasks}
            else: raise ValueError('TaskonomyDataset option transform must be a Dict[str, Callable], None, or "DEFAULT"')
                
        if 'point_info' in self.tasks and (getattr(self.transform, 'point_info', None) is None):
            self.transform['point_info'] = self._point_info_supplement

    def task_config(self, task): return task_parameters[task]

 

    ########################################################
    # Path Definitions
    ########################################################
    def get_bpv(self, path ) -> View:
        building = self._get_building_from_path(path)
        file_name = os.path.basename(path)
        try:
            lf = parse_filename( file_name )
        except:
            self.logger.error(f"Error processing path {path}")
            raise
        return View(building=building, point=lf.point, view=lf.view)

    def _get_building_from_path(self, url):
        building = url.split('/')[-2]
        return building

    def get_building_from_bpv(self, bpv):
        # self.logger.critical("Warning: get_building_from_bpv must be implemented from subclass.")
        return bpv[0]

    def _build_filename(self, building, point, view, task):
        if task in ['point_info']:  ext = 'json'
        elif task in ['fragments']: ext = 'npy'
        elif task in ['cooccurrence']: ext = 'csv'
        elif task in ['scene_metadata', 'scene_multiview_metadata']: ext = 'hdf5'
        elif task in ['mesh']:      ext = 'ply'
        else:                       ext = 'png'
        if task in ['mesh', 'cooccurrence', 'scene_metadata', 'scene_multiview_metadata']: return f"{building}.{ext}"
        else:  return f"point_{point}_view_{view}_domain_{task}.{ext}"

    def _build_path(self, building, point, view, task):
        fname = self._build_filename(building, point, view, task)
        if task in ['mesh', 'cooccurrence', 'scene_metadata', 'scene_multiview_metadata']: return os.path.join(self.data_path, task, self.dataset_name, fname)
        else: return os.path.join(self.data_path, task, self.dataset_name, building, fname)

    def _folder_in_split(self, folder, split):
        row = self.split_df.loc[ self.split_df['id']==folder]
        return (not row.empty and row.iloc[0][split] == 1)


    ########################################################
    # 3D / Point Info
    ########################################################
    def _point_info_supplement(self, point_info):
        point_info['building'] = self._get_building_from_path(point_info['path'])
        
        whitelisted = {
            k: v for (k, v) in point_info.items() if k in [
                'building',
                'camera_location',
                'camera_rotation_final',
                'camera_uuid',
                'extrinsics',
                'field_of_view_rads',
                'intrinsics',
                'path',
                'point_uuid',
                'resolution', 
                'rotation_mat', 
                'view_id',
                'nonfixated_points_in_view',
            ]
        }
        
        # Sasha: I think that I added this for generating fragments?
        whitelisted.update(
          {
            k: torch.tensor(whitelisted[k])
            for k in ['camera_rotation_final', 'camera_location', 'field_of_view_rads']
            if k in whitelisted
          }
        )
        
        return whitelisted

    def _get_cam_to_world_R_T_K(self, point_info: Dict[str, Any], building: str, point: int, view: int, device='cpu') -> List[torch.Tensor]:
        EULER_X_OFFSET_RADS = math.radians(90.0)
        location = point_info['camera_location']
        rotation = point_info['camera_rotation_final']
        fov      = point_info['field_of_view_rads']

        
        # Recover cam -> world
        ex, ey, ez = rotation
        R     = euler_angles_to_matrix(torch.tensor(
                  [(ex - EULER_X_OFFSET_RADS, -ey, -ez)],
                  dtype=torch.double, device=device), 'XZY')
        Tx, Ty, Tz = location
        T     = torch.tensor([[-Tx, Tz, Ty]], dtype=torch.double, device=device) 



        # P3D expects world -> cam
        R_inv = R.transpose(1,2)
        T_inv = -R.bmm(T.unsqueeze(-1)).squeeze(-1)
        # T_inv = -R.bmm(T.unsqueeze(-1)).squeeze(-1)
        # T_inv = T
        # R_inv = R 
        K = FoVPerspectiveCameras(device=device, R=R_inv, T=T_inv, fov=fov, degrees=False).compute_projection_matrix(znear=0.001, zfar=512.0, fov=fov, aspect_ratio=1.0, degrees=False)
        
        return dict(
          cam_to_world_R=R_inv.squeeze(0).float(),
          cam_to_world_T=T_inv.squeeze(0).float(),
          proj_K=K.squeeze(0).float(),
          proj_K_inv=K[:,:3,:3].inverse().squeeze(0).float())







#################################
#      Helper functions         #
#################################
class FileDirCache():
    def __init__(self, dirpath, overwrite_cache=False, cache_enabled=True):
        self.dirpath = os.path.abspath(dirpath)
        self.overwrite_cache = overwrite_cache
        self.cache_enabled = cache_enabled
        if self.cache_enabled: os.makedirs(self.dirpath, exist_ok=True)

    
    def get_or_eval(self, key, thunk):
        if not self.cache_enabled: return thunk()
        cache_fpath = self.keypath(key)
        if key in self:
            with open(cache_fpath, 'rb') as f:
                return pickle.load(f)
        result = thunk()
        with open(cache_fpath, 'wb') as f: pickle.dump(result, f)
        return result

    def put(self, key, value):
        if not self.cache_enabled: return value
        cache_fpath = self.keypath(key)
        with open(cache_fpath, 'wb') as f: pickle.dump(value, f)
        return value
        
    def keypath(self, key): return os.path.join(self.dirpath, f'{key}.pkl')

    def __contains__(self, key):
        return self.cache_enabled and (not self.overwrite_cache) and os.path.exists(self.keypath(key))
                                   
    def __getitem__(self, key): 
        with open(self.keypath(key), 'rb') as f: return pickle.load(f)

def parse_filename( filename ):
    p = re.match('.*point_(?P<point>\d+)_view_(?P<view>\d+)_domain_(?P<domain>\w+)', filename)
    if p is None: raise ValueError( 'Filename "{}" not matched. Must be of form point_XX_view_YY_domain_ZZ.**.'.format(filename) )
    lf = {'point': p.group('point'), 'view': p.group('view'), 'domain': p.group('domain') }
    return LabelFile(**lf)

def make_empty_like(data_dict):
    loguru.logger.critical("DeprecationWarning: Is anyone using this?")
    if not isinstance(data_dict, dict):
        if isinstance(data_dict, torch.Tensor):
            return torch.zeros_like(data_dict)
        elif isinstance(data_dict, list):
            return [make_empty_like(d) for d in data_dict]
        else:
            return type(data_dict)()
        raise NotImplementedError

    result = {}
    for k, v in data_dict.items():
        result[k] = make_empty_like(v)
    return result

def load_subfolder(subfolder_path):
    ims = []
    for fname in sorted(os.listdir(subfolder_path)):
        path = os.path.join(subfolder_path, fname)
        ims.append(os.path.join(subfolder_path, fname))
    return ims
  

#################################
#      MULTIVIEW SAMPLERS       #
#################################
class MultiviewSampler:
    def __init__(self, **kwargs):
        pass
    
    def sample(self, bpv, k=1) -> List[BPV]:
        raise NotImplementedError

    def tables_create(self, dataset, n_workers=None):
        raise NotImplementedError

    def tables_sync(self, dataset, bpv_list) -> List[BPV]:
        ''' Prunes '''
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError
    
    def load_state_dict(self, value):
        raise NotImplementedError
    
    def tables_create_and_dump(self, dataset, n_workers=None):
        self.tables_create(dataset, n_workers)
        return self.state_dict()

    @property
    def cache_key(self): return type(self).__name__


class DefaultMultiviewSampler(MultiviewSampler):
    '''
        This sampler simply uses the structure of Omnidata files to infer multiview info. 
        
        All files with point_i_view_*_domain_* have the same fixation point (point i). So
        this sampler simply uses the constructed BPV list to find all views matching BP*.
        It is fast, requires no external information, but all views orbit a single fixed 
        point. 
    '''
    
    def __init__(self, min_views, **kwargs):
        super().__init__(**kwargs)
        self.min_views = min_views

    def sample(self, bpv, k=1) -> List[BPV]:
        return random.choices(self.bpv_cooccurrence[bpv], k=k)

    def tables_create(self, dataset, n_workers=None):
        dataset.logger.info(f"Building multiview index tables...")
        self.bpv_cooccurrence = None

    def tables_sync(self, dataset, bpv_list, n_workers=None) -> List[BPV]:
        bp_to_views = defaultdict(list)
        for building, point, view in bpv_list:
            bp_to_views[(building, point)].append(view)
            
        bpv_cooccurrence = {}
        for (b, p), views in bp_to_views.items():
            if len(views) < self.min_views: continue
            for v in views:
                bpv_cooccurrence[(b,p,v)] = [(b,p,v2) for v2 in views if v2 != v]
        
        bpv_list = [bpv for bpv in bpv_list if bpv in bpv_cooccurrence]
        self.bpv_cooccurrence = bpv_cooccurrence
        return bpv_list
    
    def state_dict(self) -> Dict[str, Any]:
        return dict(
            bpv_cooccurrence=self.bpv_cooccurrence,
        )
    
    def load_state_dict(self, value):
        self.bpv_cooccurrence = value['bpv_cooccurrence']

    @property
    def cache_key(self): return type(self).__name__


class OverlapMultiviewSampler(MultiviewSampler):
    '''
        This sampler allows you to filter view sampling based on the % of pixels that
        overlap with the anchor image. 
        
        It requires having pre-generated overlap (cooccurrence) files based on fragment
        images (idx of mesh face triangles). Code for that is available in
        src.utils.fragments.?
        
        This has to be done for all pairs of images in a building. This works great for 
        Hypersim (a couple hundred frames per scene).
        
        But it works less well for Taskonomy, since there are many impates per scene. The
        resulting files can be very large (up to 40k images/scene => a billion pairs). The 
        issue is that even processing hundreds of pairs per second, computing all those
        pairs can take over a week on a single V100. Right now the data is also stored
        as a .csv, which isn't particularly efficient and can also be up to 10gb/building.
    '''
    
    def __init__(self, min_views, max_views, min_overlap_prop=0.25, **kwargs):
        super().__init__(**kwargs)
        self.min_views = min_views
        self.max_views = max_views
        self.min_overlap_prop = min_overlap_prop

    def sample(self, bpv, k=1):
        result = random.choices(self.bpv_cooccurrence[bpv], k=k)
        return [v[:3] for v in result] # drop the prop_shared

    def tables_create(self, dataset, n_workers=None) -> List[BPV]:
        ''' Returns: pruned BPV list '''
        dataset.logger.info(f"Building multiview index tables...")

        # Load all the files to memory
        cooccurrence_dirpath = os.path.join(dataset.data_path, 'cooccurrence', dataset.dataset_name)
        filenames            = [fname for fname in glob.glob(os.path.join(cooccurrence_dirpath, '*')) if fname.endswith('.csv')] 
        # bpv_cooccurrences = [_building_cooccurrences_thunk(fname) for fname in tqdm(filenames)]
        bpv_cooccurrences    = process_map(ft.partial(_building_cooccurrences_thunk, max_views=self.max_views, min_overlap_prop=self.min_overlap_prop),
                 filenames, # Process map hangs here
                 desc        = f'Building cooccurrence tables ({n_workers} workers)',
                 max_workers = n_workers,
                 chunksize   = min(1, len(filenames) // n_workers + 1))
        self.bpv_cooccurrence = ft.reduce(lambda a, b: {**a, **b}, bpv_cooccurrences)

    def tables_sync(self, dataset, bpv_list, n_workers=None) -> List[BPV]:
        # Now sync with BPV list
        # dataset.logger.info("Filtering out missing BPVs from cooccurrence tables...")
        full_bpv_cooccurrence = self.bpv_cooccurrence
        bpv_set = set(bpv_list)
        new_bpv_list, missing_cooccurrences = [], []
        missing_bpv = []
        self.bpv_cooccurrence = {}
        for bpv in bpv_list:
            good_dests = [dest for dest in full_bpv_cooccurrence.get(bpv, []) if dest[:3] in full_bpv_cooccurrence and dest[:3] in bpv_set]
            if len(good_dests) < self.min_views:
                # missing_cooccurrences.append(bpv)
                if bpv in full_bpv_cooccurrence: missing_cooccurrences.append(bpv)
                else: missing_bpv.append(bpv)
            else: 
                self.bpv_cooccurrence[bpv] = good_dests
                new_bpv_list.append(bpv)
        if len(missing_cooccurrences) > 0: self.logger.warning(f'\tMissing cooccurence information for some views ({len(missing_bpv)} missing bpv) ({len(missing_cooccurrences)} missing cooccurrence ). Filtering down.')
        return new_bpv_list

    def state_dict(self) -> Dict[str, Any]:
        return dict(
            bpv_cooccurrence=self.bpv_cooccurrence,
        )

    def load_state_dict(self, value):
        for k, v in value.items():
            setattr(self, k, v)

    @property
    def cache_key(self): return f'{type(self).__name__}_min_{self.min_views}_max_{self.max_views}_overlap_{self.min_overlap_prop}'

def _building_cooccurrences_thunk(fpath, max_views, min_overlap_prop, bpv_set=None):
    csv    = pd.read_csv(fpath)
    csv    = csv[csv['valid_and_shared_prop'] >= min_overlap_prop]
    keys   = ['src.building', 'src.point', 'src.view']
    metric = 'valid_and_shared_prop'
    subset = csv.groupby(keys,group_keys=True).apply(lambda grp:grp.nlargest(n=max_views, columns=metric).sort_values(keys + [metric],ascending=False))
    bpv_cooccurrence = {}
    for row in subset.iloc:
        src = tuple(str(row[k]) for k in keys)
        new_dst = (row['dst.building'], str(row['dst.point']), str(row['dst.view']), row['valid_and_shared_prop'])
        bpv_cooccurrence[src] = bpv_cooccurrence.get(src, []) + [new_dst]
    return bpv_cooccurrence




class CenterVisibleMultiviewSampler(MultiviewSampler):
    '''
        This sampler builds a graph of each Omnidata scene:
            Nodes: are each view (camera + target point)
            Edges: point to A to B when the center of view B is visible from A
        
        We can then build the graph several ways:
            sampled_camera type:
                SAME: Same camera locaiton
                FIXATED: Same camera target point
                DIFFERENT: Neither of the above
        
                
    '''
    def __init__(self,
            sampled_camera_type,
            path_length=1,
            max_tries=10,
            backoff_order=('SAME', 'FIXATED', 'DIFFERENT'),
            sampled_camera_knn=1,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.sampled_camera_knn = sampled_camera_knn
        self.new_camera_type = sampled_camera_type
        self.max_tries = max_tries
        self.backoff_order = backoff_order
        self.path_length = path_length
        self.scene_metadata = {}
        self.scene_mv_metadata = {}

    def sample(self, bpv, k=1, path_length=None, new_camera_type=None) -> List[BPV]:
        new_camera_type = self.new_camera_type if new_camera_type is None else new_camera_type
        path_length  = self.path_length if path_length is None else path_length
        to_return = []
        # self.backoff_order = ['SAME', 'FIXATED', 'DIFFERENT']
        for _k in range(k):
            new_bpv = bpv
            for _ in range(path_length):
                new_bpv =  CenterVisibleMultiviewSampler_one_hop(
                        new_bpv,
                        bm = self.scene_metadata[self.building_to_key[bpv[0]]],
                        bmm = self.scene_mv_metadata[self.building_to_key[bpv[0]]],
                        new_camera_type = new_camera_type,
                        max_tries = self.max_tries,
                        backoff_order = self.backoff_order,
                        sampled_camera_knn = self.sampled_camera_knn,
                    )
            to_return.append(new_bpv)

        return to_return

    def tables_create(self, dataset, n_workers=None):
        '''  '''
        pass


    def tables_sync(self, dataset, bpv_list, n_workers=1) -> List[BPV]:
        ''' Prunes'''
        dataset.logger.info(f"Building multiview index tables...")
        if n_workers > 40: n_workers = 40
        # buildings_sorted = sorted(bm.buildings)
        # bmm_buildings_sorted = sorted(bmm.buildings)
        # assert buildings_sorted == bmm_buildings_sorted, f"{buildings_sorted} != {bmm_buildings_sorted}"
        
        # Load all the files to memory
        metadata_dirpath  = os.path.join(dataset.data_path, 'scene_metadata', dataset.dataset_name)
        filenames         = [fname for fname in glob.glob(os.path.join(metadata_dirpath, '*')) if fname.endswith('.hdf5')] 
        scene_metadata    = thread_map(ft.partial(BuildingMetadata.read_hdf5, bpv_list=bpv_list),
                 filenames, 
                 desc        = f'Loading scene metadata ({n_workers} workers)',
                 max_workers = n_workers,
                 chunksize   = max(1, len(filenames) // n_workers + 1)
                , leave=False)
        self.scene_metadata = {
            os.path.basename(fname).replace('.hdf5',''): metadata
            for fname, metadata in zip(filenames, scene_metadata)
        }

        mv_metadata_dirpath  = os.path.join(dataset.data_path, 'scene_multiview_metadata', dataset.dataset_name)
        filenames         = [fname for fname in glob.glob(os.path.join(mv_metadata_dirpath, '*')) if fname.endswith('.hdf5')] 
        scene_mv_metadata = thread_map(ft.partial(BuildingMultiviewMetadata.read_hdf5, bpv_list=bpv_list),
                 filenames,
                 desc        = f'Loading scene multiview metadata ({n_workers} workers)',
                 max_workers = n_workers,
                 chunksize   = max(1, len(filenames) // n_workers + 1))    
        
        self.scene_mv_metadata = {
            os.path.basename(fname).replace('.hdf5',''): metadata
            for fname, metadata in zip(filenames, scene_mv_metadata)
        }


        self._make_building_to_key()
        
        new_bpv_list = []
        for bpv in bpv_list:
            failed = False
            bkey      = self.building_to_key[bpv[0]]
            bm, bmm   = self.scene_metadata[bkey], self.scene_mv_metadata[bkey]
            bm_bpv_enc, bmm_bpv_enc = bm.encode_bpv(bpv), bmm.encode_bpv(bpv)
            in_bm  = (bm_bpv_enc in bm.BPV_to_camera_idx)
            in_bmm = (bmm_bpv_enc in bmm.bpv_to_all_visible_bp)
            # If bpv not found in bm or bmm, remove from everything.
            if (not in_bmm) or (not in_bm):
                if in_bm:  bm.remove_bpv(bm_bpv_enc)
                if in_bmm: bmm.remove_bpv(bmm_bpv_enc)
            else:
                new_bpv_list.append(bpv)


        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)

            for k, _bmm in self.scene_mv_metadata.items():
                _bm = self.scene_metadata[k]
                _bmm.bp_to_all_visible_bpc = {}
                for bp, _bpvs in _bmm.bp_to_all_visible_bpv.items():
                    _bmm.bp_to_all_visible_bpc[tuple(bp)] = np.array([
                        (_bpv[0], _bpv[1], _bm.BPV_to_camera_idx[tuple(_bpv)]) for _bpv in _bpvs
                    ])
                _bmm.bp_to_all_visible_bpc = pd.Series(_bmm.bp_to_all_visible_bpc)

        return new_bpv_list

    def _make_building_to_key(self):
        self.building_to_key = {}
        for k, bm in self.scene_metadata.items():
            for building in bm.buildings:
                self.building_to_key[building] = k
                
    def state_dict(self) -> Dict[str, Any]:
        return dict(
            scene_metadata = self.scene_metadata,
            scene_mv_metadata = self.scene_mv_metadata
        )
            
    def load_state_dict(self, value):
        for k, v in value.items():
            setattr(self, k, v)
        self._make_building_to_key()

    def tables_create_and_dump(self, dataset, n_workers=None):
        self.tables_create(dataset, n_workers)
        return self.tables_dump()

    
def _check_bpv_in_multiview_metadata(self, metadata, dataset, bpv):
    bm = metadata[dataset.get_building_from_bpv(bpv)]
    return (bm.encode_bpv(bpv) in bm.BPV_to_camera_idx)
    # bmm = self.scene_mv_metadata[dataset.get_building_from_bpv(bpv)]
# and  (bmm.encode_bpv(bpv) in bmm.bpv_to_all_visible_bp)

def get_scene_and_multiview_metadata(self, scene_name):
    scene_file = self._build_path(scene_name, None, None, None, 'scene_metadata')
    bm = BuildingMetadata.read_hdf5(scene_file)
    scene_multiview_file = self._build_path(scene_name, None, None, None, 'scene_multiview_metadata')
    bmm = BuildingMultiviewMetadata.read_hdf5(scene_multiview_file)
    return bm, bmm


def CenterVisibleMultiviewSampler_one_hop(
        bpv,
        bm: BuildingMetadata,
        bmm: BuildingMultiviewMetadata,
        new_camera_type: str='DIFFERENT',
        max_tries: int=10,
        backoff: bool=True,
        backoff_order=('SAME', 'FIXATED', 'DIFFERENT'),
        sampled_camera_knn: int = None,
        camera_knn_exclusion_radius: float=1e-6,
    ) -> Tuple[str]:
    '''
        Define a Point as the (building, point) tuple so they uniquely identify the BP
        
        Bipartite graph of scene:
            Cameras <-> Points
        
        Pair of (Camera, Point) will uniquely identify an image in the dataset. (bijection between BPC and BPV)
    '''
    if new_camera_type.upper() == 'BACKOFF': new_camera_type = backoff_order[-1]
    b,p,v  = bpv
    new_camera_type = new_camera_type.upper()
    def do_backoff_or_raise(on_failure_message):
        if backoff: 
            camera_type_idx = backoff_order.index(new_camera_type)
            if camera_type_idx == 0: return bpv # raise LookupError(on_failure_message)
            return CenterVisibleMultiviewSampler_one_hop(
                    bpv,
                    bm = bm,
                    bmm = bmm,
                    new_camera_type = backoff_order[camera_type_idx - 1],
                    max_tries = max_tries,
                    backoff = backoff,
                    backoff_order=backoff_order,
                    sampled_camera_knn = sampled_camera_knn,
                )
        if new_camera_type in ['SAME', 'ANY']: return bpv
        raise LookupError(on_failure_message)
    
    # Points in view
    bps_all = bmm.bpv_to_all_visible_bp[(bm.B_to_idx[b], int(p), int(v))]
    if len(bps_all) == 0: return do_backoff_or_raise(f'Could not find ANY BPs in view of BPV {bpv}')
    
    # Get all cameras path length 2 away (biparitite graph w/ cameras on one side and points on other)
    bpv_np  = (bm.B_to_idx[b], int(p), int(v))
    cam     = bm.BPV_to_camera_idx[bpv_np]
    bpcs = []
    # bps, neighborhood_cams = [], []
    for bp in bps_all:
        if np.all(bp == bpv_np[:2]) and new_camera_type in ['DIFFERENT']: continue 
        if np.all(bp != bpv_np[:2]) and new_camera_type in ['FIXATED']: continue 
        _bpcs = bmm.bp_to_all_visible_bpc.get(tuple(bp), np.array([], dtype=int))
        if len(_bpcs) == 0: continue
        bpcs.append(_bpcs)

    
    if len(bpcs) == 0: return do_backoff_or_raise(f'Could not find any valid BPs in view of BPV {bpv}')
    bpcs = np.concatenate(bpcs, axis=0)
    neighborhood_cams = bpcs[:, -1]

    # If using same camera, just pick another BP in view
    if new_camera_type == 'SAME':
        viable = bpcs[neighborhood_cams == cam]
        if len(viable) == 0: return do_backoff_or_raise(f'Could not find any {new_camera_type} BPs in view of BPV {bpv}')
        bpc = tuple(random.choice(viable))
        return (b, str(bpc[1]), str(bm.BPC_to_view_idx[bpc]))

    # Select only KNN (by camera location)
    if sampled_camera_knn is not None and sampled_camera_knn > 0:
        diff_cam  = (neighborhood_cams != cam)
        neighborhood_cams_uniq = np.unique(neighborhood_cams[diff_cam])
        if len(neighborhood_cams_uniq) == 0: return do_backoff_or_raise(f'Could not find any {new_camera_type} BPs in view of BPV {bpv}')
        dists   = np.sum((bm.camera_set.locs[cam][np.newaxis, :] - bm.camera_set.locs[neighborhood_cams_uniq])**2, axis=-1)
        indices = np.argsort(dists)[:sampled_camera_knn]
        cutoff  = dists[indices[min(sampled_camera_knn, len(indices)-1)]]

        dists_all = np.sum((bm.camera_set.locs[cam][np.newaxis, :] - bm.camera_set.locs[neighborhood_cams])**2, axis=-1)
        keep      = (dists_all <= cutoff) & diff_cam
        neighborhood_cams = neighborhood_cams[keep]
        bpcs = bpcs[keep]
        # print(f'Keeping {len(indices)} / {len(neighborhood_cams)} cameras (dist <= {cutoff}): ({len(bps)} / {len(bps_all)}) points')
        # raise NotImplementedError
    
    # Select one of the candidate BPVs
    bpc_np  = random.choice(bpcs)
    new_view = bm.BPC_to_view_idx[tuple(bpc_np)]
    new_bpv =  (bm.buildings[bpc_np[0]], str(bpc_np[1]), str(new_view))
    # idx      = random.randrange(len(neighborhood_cams))
    # print(bp[0], bp[1], new_cam, idx, len(bps), type(bp[1]))
    # print(f'{bpv} (cam {cam}) -> {new_bpv} (cam {bpc_np[-1]})')
    return new_bpv #(bm.buildings[bp[0]], str(bp[1]), str(new_view))
   
# def CenterVisibleMultiviewSampler_one_hop(
#         bpv,
#         bm: BuildingMetadata,
#         bmm: BuildingMultiviewMetadata,
#         new_camera_type: str='DIFFERENT',
#         max_tries: int=10,
#         backoff: bool=True,
#         backoff_order=('SAME', 'FIXATED', 'DIFFERENT'),
#         sampled_camera_knn: int = 10,
#     ) -> Tuple[str]:
#     if new_camera_type.upper() == 'BACKOFF': new_camera_type = backoff_order[-1]
#     b,p,v  = bpv
#     new_camera_type = new_camera_type.upper()
#     # Sample from different camera
#     bpv_np = (bmm.B_to_idx[b], int(p), int(v))
#     cam    = bm.BPV_to_camera_idx[bpv_np]
#     bps    = bmm.bpv_to_all_visible_bp[(bmm.B_to_idx[b], int(p), int(v))]
#     for i in range(max_tries):
#         if len(bps) == 0: break
#         # Sample new point
#         bp = random.choice(bps) if new_camera_type != 'FIXATED' else (bpv_np[:2])
#         if tuple(bp) not in bm.BP_to_all_visible_cameras: continue
#         # Just get view of this camera fixated at new point
#         if new_camera_type == 'SAME':
#             if (bp[0], bp[1], cam) not in bm.BPC_to_view_idx: continue
#             view = bm.BPC_to_view_idx[(bp[0], bp[1], cam)]
#             return (b,  str(bp[1]), str(view))

#         # Find new camera at new point
#         visible_cams = bm.BP_to_all_visible_cameras[tuple(bp)]
#         if new_camera_type in ['DIFFERENT', 'FIXATED']:
#             visible_cams = [c for c in visible_cams if c != cam]
#         elif new_camera_type == 'ANY':
#             visble_cams = list(visible_cams)
#         else:
#             raise NotImplementedError(f'Unknown camera type: {new_camera_type}')
#         if   len(visible_cams) == 0: continue
#         new_cam  = random.choice(visible_cams)
#         new_view = bm.BPC_to_view_idx[(bp[0], bp[1], new_cam)]
#         return (bm.buildings[bp[0]], str(bp[1]), str(new_view))
    
#     if not backoff: raise LookupError(f'Could not find satisfactory sample after {max_tries} tries')
#     camera_type_idx = backoff_order.index(new_camera_type)
#     if camera_type_idx == 0: return bpv # Just return the initial view
#     return CenterVisibleMultiviewSampler_one_hop(
#                 bpv,
#                 bm = bm,
#                 bmm = bmm,
#                 new_camera_type = backoff_order[camera_type_idx - 1],
#                 max_tries = max_tries,
#                 backoff = backoff,
#                 backoff_order=backoff_order,
#             )



MULTIVIEW_SAMPLING_METHODS = dict(
    FILENAME       = DefaultMultiviewSampler, 
    SHARED_PIXELS  = OverlapMultiviewSampler,
    CENTER_VISIBLE = CenterVisibleMultiviewSampler,
)
