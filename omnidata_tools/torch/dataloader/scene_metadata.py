'''
  This file contains 
  
    BuildingMetadata: 
      Information about the points + views in a scene
      
    CameraSet:
      pass
      
    BuildingMultiviewMetadata:
      Information about view overlap
'''

from collections import defaultdict, namedtuple
from functools import cached_property
import h5py
import numpy as np
import pandas as pd
PointView = namedtuple('PointView', ['point', 'view'])
BPV = namedtuple('BPV', ['building', 'point', 'view'])
BPC = namedtuple('BPC', ['building', 'point', 'camera'])
BP  = namedtuple('BP', ['building', 'point'])


import bisect

class KeyifyList(object):
    ''' In Python 3.10 we can add a key.
        I don't feel like upgrading all of my dependencies and fixing the resulting bugs, so here's a kludge.
        From https://gist.github.com/ericremoreynolds/2d80300dabc70eebc790
    '''
    def __init__(self, inner, key):
        self.inner = inner
        self.key = key

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, k):
        return self.key(self.inner[k])

def get_bpv_chunk_from_sorted_bpv_list(bpv_list, bm):
    if not isinstance(bm.buildings, list): raise ArgumentError(type(bm.buildings))
    buildings_sorted = sorted(bm.buildings)
    # bmm_buildings_sorted = sorted(bmm.buildings)
    # assert buildings_sorted == bmm_buildings_sorted, f"{buildings_sorted} != {bmm_buildings_sorted}"
    bpv_list_keyed = KeyifyList(bpv_list, lambda x: x[0])
    bpv_list_new = []
    for building in buildings_sorted:
        left  = bisect.bisect_left(bpv_list_keyed, building)
        right = bisect.bisect_right(bpv_list_keyed, building)
        bpv_list_new.extend(bpv_list[left:right])
    return bpv_list_new

  
#######################################
#            Single View              #
#######################################
class BuildingMetadata():
    '''
        We have the following metadata:
        
        A list of camera idxs, and their 3D locations:
            The camera idxs are unique within buildings.
        
        Dict of points + views -> camera ids:
            The combination of points + views are unique withing buildings
            
        camera_to_all_visible_BP
    '''

    def __init__(self):
        self.camera_set = CameraSet()
        # self.pv_set = set()
        self.camera_to_all_visible_BP   = defaultdict(set)
        self.BP_to_all_visible_cameras  = defaultdict(set)
        self.BPV_to_camera_idx          = {}
        self.BPC_to_view_idx            = {}
        self.B_to_idx                   = {}

    def freeze(self):
        self.camera_to_all_visible_BP = pd.Series({k: np.array(list(v)) for k, v in self.camera_to_all_visible_BP.items()})
        self.BP_to_all_visible_cameras = pd.Series({k: np.array(list(v)) for k, v in self.BP_to_all_visible_cameras.items()})
        self.BPV_to_camera_idx = pd.Series(self.BPV_to_camera_idx)
        self.BPC_to_view_idx = pd.Series(self.BPC_to_view_idx)
        self.B_to_idx = pd.Series(self.B_to_idx)
        # self.camera_to_all_visible_BP = {k: np.array(list(v)) for k, v in self.camera_to_all_visible_BP.items()}
        # self.BP_to_all_visible_cameras = {k: np.array(list(v)) for k, v in self.BP_to_all_visible_cameras.items()}
        # self.BPV_to_camera_idx = self.BPV_to_camera_idx
        # self.BPC_to_view_idx = self.BPC_to_view_idx
        # self.B_to_idx = self.B_to_idx
        self.camera_set.locs = np.array(self.camera_set.locs)

    def add_point_info(self, point_info):
        building, point, view = point_info['building'], int(point_info['point']), int(point_info['view'])
        if building not in self.B_to_idx: self.B_to_idx[building] = len(self.B_to_idx)
        cam_idx = self.camera_set.add(get_cam_loc(point_info))
        self.camera_to_all_visible_BP[cam_idx].add(point)
        self.BP_to_all_visible_cameras[(building, point)].add(cam_idx)
        self.BPC_to_view_idx[BPC(building=building, point=point, camera=cam_idx)] = view
        self.BPV_to_camera_idx[BPV(building=building, point=point, view=view)]    = cam_idx


    def remove_bpv(self, bpv):
        cam_idx = self.BPV_to_camera_idx[bpv]
        bpc = (bpv[0], bpv[1], cam_idx)
        self.BP_to_all_visible_cameras[bpv[:2]].remove(cam_idx)
        self.BPV_to_camera_idx.pop(bpv)
        self.BPC_to_view_idx.pop(bpc)

    def encode_bpv(self, bpv):
        return (self.B_to_idx[bpv[0]], int(bpv[1]), int(bpv[2]))
        
    def save_hdf5(self, fpath):
        bpvc      = np.array([(self.B_to_idx[bpv[0]], bpv[1], bpv[2], c) for bpv, c in self.BPV_to_camera_idx.items()])
        cam_locs  = np.array([np.array(t) for t in self.camera_set.locs])
        with h5py.File(fpath, "w") as f:
            cams = f.create_dataset("camera_locs", dtype='f', data=cam_locs)
            dset = f.create_dataset("building_points_views_cameras", dtype='i', data=bpvc)
            b_id = f.create_group("building_ids")
            for building_name, idx in self.B_to_idx.items():
                b_id.attrs[building_name] = idx

    def load_hdf5(self, fpath, bpv_list=None):
        with h5py.File(fpath, "r") as f:
            # Load cameras
            cam_loc = np.array(f["camera_locs"])
            self.camera_set.locs = np.array([l for l in cam_loc])
            
            # Load building IDs
            buildings = [None] * len(f['building_ids'].attrs)
            for building_name, idx in f['building_ids'].attrs.items():
                self.B_to_idx[building_name] = idx
                buildings[idx] = building_name
            self.buildings = buildings

            # Load BPVCs
            bpvc     = np.array(f["building_points_views_cameras"])

        if bpv_list is not None:
            bpv_set = set([
                (self.B_to_idx[bpv[0]], int(bpv[1]), int(bpv[2]))
                for bpv in get_bpv_chunk_from_sorted_bpv_list(bpv_list, self)
            ])
            if len(bpv_set) == 0: return

        for b_idx, point, view, cam_idx in bpvc:
            bpv = (b_idx, point, view)
            if (bpv_list is not None) and (bpv not in bpv_set): continue
            # building = buildings[b_idx]
            self.BPV_to_camera_idx[BPV(building=b_idx, point=point, view=view)]    = cam_idx
            self.BPC_to_view_idx[BPC(building=b_idx, point=point, camera=cam_idx)] = view
            self.camera_to_all_visible_BP[cam_idx].add((b_idx, point))
            self.BP_to_all_visible_cameras[(b_idx, point)].add(cam_idx)

        self.BP_to_all_visible_cameras = dict(self.BP_to_all_visible_cameras)
        self.camera_to_all_visible_BP = dict(self.camera_to_all_visible_BP)

            
    def __len__(self): return len(self.BPV_to_camera_idx)

    @classmethod
    def read_hdf5(cls, fpath, *args, **kwargs):
        res = cls()
        res.load_hdf5(fpath, *args, **kwargs)
        return res

    
    
class CameraSet():
    def __init__(self, atol=1e-2, cam_locs=None):
        self.atol = atol
        self.locs = [] if cam_locs is None else cam_locs
        self._frozen = True
        
    def loc_to_id(self, loc):
        for i, _loc in enumerate(self.locs):
            if torch.allclose(loc, _loc, atol=self.atol, equal_nan=False):
                return i
        return None

    def add(self, loc):
        if self._frozen: raise RuntimeError("Attempting to add to a frozen CameraSet")
        _id = self.loc_to_id(loc)
        if _id is not None: return _id
        self.locs.append(loc)
        return len(self.locs) - 1
        
    def loc(self, _id):
        return self.locs[_id]

    def __len__(self): return len(self.locs)

    def pdist(self):
        return torch.nn.functional.pdist(self.locs, p=2).min()

    def freeze(self): 
        self._frozen = True
        self.locs = torch.stack(locs, dim=0)

def get_cam_loc(point_info):
    return point_info['cam_to_world_R'] @ point_info['cam_to_world_T']

  

#######################################
#             Multiview               #
#######################################
from collections import defaultdict, Counter, namedtuple
import numpy as np

class BuildingMultiviewMetadata():
    '''
        We have the following metadata:
        
        bpv_to_all_visible_bp
        bp_to_all_visible_bpv
        B_to_idx:
    '''
    def __init__(self):
        self.bpv_to_all_visible_bp = {}
        self.bp_to_all_visible_bpv = {}
        self.B_to_idx              = {}
    
    def freeze(self):
        self.bpv_to_all_visible_bp = pd.Series({k: np.array(list(v)) for k, v in self.bpv_to_all_visible_bp.items()})
        self.bp_to_all_visible_bpv = pd.Series({k: np.array(list(v)) for k, v in self.bp_to_all_visible_bpv.items()})
        if hasattr(self, 'bp_to_all_visible_bpc'):
            self.bp_to_all_visible_bpc = pd.Series({k: np.array(list(v)) for k, v in self.bp_to_all_visible_bpc.items()})
        self.B_to_idx = pd.Series(self.B_to_idx)

    def remove_bpv(self, bpv):
        bps = self.bpv_to_all_visible_bp.pop(bpv)
        for bp in bps:
            bp = int(bp[0]), int(bp[1])
            self.bp_to_all_visible_bpv[bp].remove(bpv)

    def encode_bpv(self, bpv):
        return (self.B_to_idx[bpv[0]], int(bpv[1]), int(bpv[2]))
        
    def save_hdf5(self, fpath):
        with h5py.File(fpath, "w") as f:
            bpv_to_visible_bp = f.create_group("bpv_to_all_visible_bp")
            for bpv, bp in self.bpv_to_all_visible_bp.items():
                bpv_to_visible_bp.create_dataset(f"({bpv.building}, {bpv.point}, {bpv.view})", data=np.array(bp))

            b_id = f.create_group("building_ids")
            for building_name, idx in self.B_to_idx.items():
                b_id.attrs[building_name] = idx

    def load_hdf5(self, fpath, bpv_list=None):
        with h5py.File(fpath, "r") as f:
            # Load building IDs
            self.buildings = [None] * len(f['building_ids'].attrs)
            for building_name, idx in f['building_ids'].attrs.items():
                self.B_to_idx[building_name] = idx
                self.buildings[idx] = building_name
            
            if bpv_list is not None:
                bpv_set = set([
                    (self.B_to_idx[bpv[0]], int(bpv[1]), int(bpv[2]))
                    for bpv in get_bpv_chunk_from_sorted_bpv_list(bpv_list, self)
                ])
                if len(bpv_set) == 0: return

            # Load bpv -> bp
            for bpv_str, bps in f['bpv_to_all_visible_bp'].items():
                bpv = eval(bpv_str)
                if (bpv_list is not None) and (bpv not in bpv_set): continue
                self.bpv_to_all_visible_bp[bpv] = np.array(bps)

            for bpv, bps in self.bpv_to_all_visible_bp.items():
                for bp in bps:
                    bp = (int(bp[0]), int(bp[1]))
                    if bp not in self.bp_to_all_visible_bpv: self.bp_to_all_visible_bpv[bp] = set()
                    self.bp_to_all_visible_bpv[bp].add(bpv)
            

    @classmethod
    def read_hdf5(cls, fpath, *args, **kwargs):
        res = cls()
        res.load_hdf5(fpath, *args, **kwargs)
        return res

    def compute_from_point_info(self, ds):
        self.B_to_idx = self.get_buildings_to_idx(ds)
        for bpv, datapoint in ds.items():
            pi = datapoint['positive']['point_info'][0]
            visible_points = pi['nonfixated_points_in_view']
            bps = np.stack([[0] * len(visible_points), visible_points], axis=1)
            self.bpv_to_all_visible_bp[BPV(self.B_to_idx[bpv.building], bpv.point, bpv.view)] = bps
        self.bp_to_all_visible_bpv = defaultdict(list)
        for bpv, bps in tqdm(self.bpv_to_all_visible_bp.items()):
            for bp in bps:
                self.bp_to_all_visible_bpv[BP(bp[0].item(), bp[1].item())].append(BPV(bpv.building, bpv.point, bpv.view))

    # These methods are how the multiview information is computed. Requires a DatasetInMem as ds
    def compute_from_frags(self, ds, device='cpu', store_on_device=False):
        return_device = device if store_on_device else 'cpu'
        self.B_to_idx = self.get_buildings_to_idx(ds)
        points_to_center_frag = self.get_center_frag(ds, buildings_to_idx=self.B_to_idx)
        unique_frags  = self.compute_unique_fragments(ds, device=device, store_on_device=store_on_device)
        points_in_view = self.compute_points_in_view(unique_frags, points_to_center_frag, compute_device=device, return_device=return_device)
        visible_bpv = defaultdict(list)
        for bpv, bps in tqdm(points_in_view.items()):
            for bp in bps:
                if (bp[0].item(), bp[1].item()) == (self.B_to_idx[bpv.building], int(bpv.point)): continue
                visible_bpv[BP(bp[0].item(), bp[1].item())].append(BPV(self.B_to_idx[bpv.building], int(bpv.point), int(bpv.view)))
        self.bp_to_all_visible_bpv = visible_bpv
        self.bpv_to_all_visible_bp = {}
        for bpv, bps in points_in_view.items():
            keep = ~(bps == torch.tensor((self.B_to_idx[bpv.building], int(bpv.point)))).all(dim=1)
            bps = bps[keep]
            self.bpv_to_all_visible_bp[BPV(self.B_to_idx[bpv.building], int(bpv.point), int(bpv.view))] = bps

    @classmethod
    def get_buildings_to_idx(cls, ds):
        b_to_idx = {}
        for bpv, pi in ds.items(): 
            building = pi['positive']['building']
            if building not in b_to_idx: b_to_idx[building] = len(b_to_idx)
        return b_to_idx

    @classmethod
    def get_center_frag(cls, ds, half_window_size=1, buildings_to_idx=None, desc='Computing center fragments'):
        if buildings_to_idx is None: buildings_to_idx = get_buildings_to_idx(ds)
        points_to_center_frag = {}
        for bpv, pi in tqdm(ds.items(), total=len(ds)):
            pi = pi['positive']
            frags = pi['fragments'][0]
            half_window = half_window_size
            while True:
                center_frag = frags[frags.shape[0] // 2 - half_window:frags.shape[0] // 2 + half_window, frags.shape[1] // 2- half_window:frags.shape[1] // 2 + half_window:] 
                nonzero = center_frag[center_frag > 0]
                if len(nonzero) == 0:
                    half_window += 1
                else:
                    center_frag = torch.mode(nonzero).values.item()
                    break
            points_to_center_frag[(int(buildings_to_idx[pi['building']]), int(pi['point']), int(pi['view']))] = center_frag
        return points_to_center_frag

    @classmethod
    def compute_unique_fragments(cls, ds, device='cpu', store_on_device=False):
        unique = {}
        # unique = {k: v['positive']['fragments'].to(device, non_blocking=True).unique() for k, v in tqdm(sds.dataset.items(), desc='Generating fragment signatures')}
        for k, v in tqdm(ds.items(), desc='Generating fragment signatures', total=len(ds)):
            frags = v['positive']['fragments'].to(device).unique()
            unique[k] = frags if store_on_device else frags.cpu().pin_memory()
        return unique

    @classmethod
    def compute_points_in_view(cls, unique_frags, points_to_center_frag, compute_device, return_device='cpu'):
        center_frag_pv      = torch.tensor(list(points_to_center_frag.keys()), dtype=torch.int, device=compute_device)
        center_frags   = torch.tensor(list(points_to_center_frag.values()), dtype=torch.long, device=compute_device)
        result = {}
        for bpv, frags in tqdm(unique_frags.items()):
            frags = frags.to(compute_device)
            pv_in_view = center_frag_pv[[v in frags for v in center_frags]]
            result[bpv] = pv_in_view[:,:2].unique(dim=0).to(return_device)
            # p_in_view  = pv_in_view[:,0].unique().to(return_device)
        return result