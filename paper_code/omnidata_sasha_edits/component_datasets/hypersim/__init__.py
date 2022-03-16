
import json
import multiprocessing as mp
import os
from   pandas import read_csv
from   PIL import Image
import torch
from   torchvision import transforms
import tqdm
from   tqdm.contrib.concurrent import process_map  # or thread_map

from ...segment_instance import random_colors
from ...splits import get_splits
from ...omnidata_dataset import OmnidataDataset, load_subfolder

###############################################################################
# Split info:
# Exports:
#   e.g. flat_split_to_spaces['tiny-train'] -> List[str] building names
###############################################################################
split_file = os.path.join(os.path.dirname(__file__), 'train_val_test_hypersim.csv')
flat_split_to_spaces = get_splits(split_file)

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



###############################################################################
# Make dataset 
###############################################################################
def make_task_dataset_new_split(self, task):
    dir = os.path.join(self.data_path, task)
    folders = flat_split_to_spaces[f'{self.split}']
    if task == 'segment_semantic': task = 'semantic_hdf5'
    #  folders are building names. 
    images = []
    dir = os.path.expanduser(dir)
    if not os.path.isdir(dir):
        assert "bad directory"

    for folder in folders:
        taskonomized_path = os.path.join(dir, folder, 'taskonomized')
        for camera in os.listdir(taskonomized_path):
            if not camera.startswith('cam'):
                continue
            # filter out bad points from filtered_points.json
            with open(os.path.join(taskonomized_path, camera, 'filtered_points.json')) as json_file:
                bad_points = json.load(json_file)
            folder_path = os.path.join(taskonomized_path, camera, task)
            for fname in sorted(os.listdir(folder_path)):
                point = fname.split('_')[1]
                if point not in bad_points:
                    path = os.path.join(folder_path, fname)
                    images.append(path)

    return images



def semseg_remap_inplace(res):
    res2 = res.clone() 
    labels = torch.unique(res)
    for old_label in labels:
        if old_label == -1 or old_label == 255: continue
        res[res2 == old_label] = CLASS_LABEL_TRANSFORM[old_label]
    return res

class HypersimDataset(OmnidataDataset):

    def make_task_dataset(self, task):
        hypersim_orig_split_file = os.path.join(os.path.dirname(__file__), f'{self.split}_hypersim_orig.csv')
        df = read_csv(hypersim_orig_split_file)

        if task == 'segment_semantic': task = 'semantic_hdf5'
        #  folders are building names. 
        dirpath = self.data_path
        images = []
        dirpath = os.path.expanduser(dirpath)
        if not os.path.isdir(dirpath):
            assert "bad directory"

        folders = [scene for scene in set(df['scene_name'].tolist()) if scene in os.listdir(dirpath)]
        for folder in tqdm.tqdm(folders, desc=f'Loading {task} paths'):
            scene = df.loc[df['scene_name']==folder]
            taskonomized_path = os.path.join(dirpath, folder, 'taskonomized')
            for camera in os.listdir(taskonomized_path):
                if not camera.startswith('cam'):
                    continue
                scene_cam = scene.loc[scene['camera_name']==camera]
                # filter out bad points from filtered_points.json
                with open(os.path.join(taskonomized_path, camera, 'filtered_points.json')) as json_file:
                    bad_points = json.load(json_file)
                folder_path = os.path.join(taskonomized_path, camera, task)
                for fname in sorted(os.listdir(folder_path)):
                    point = fname.split('_')[1]
                    if point in bad_points: continue

                    row = scene_cam.loc[scene_cam['frame_id']==int(point)]
                    if not row.empty and row.iloc[0]['included_in_public_release'] and row.iloc[0]['split_partition_name'] == self.split:
                        path = os.path.join(folder_path, fname)
                        images.append(path)

        return images

    def _get_building(self, url):
        building = url.split('/')[-5] + '-' + url.split('/')[-3]
        return building

    def __init__(self, options: OmnidataDataset.Options):
        if options.image_size is None:
            options.image_size = 512
        super().__init__(options)
        for task, _transform in self.transform.items():
            resize_method = Image.BILINEAR if task in ['rgb'] else Image.NEAREST
            _new_transform = transforms.Compose([
                transforms.Resize(self.image_size, resize_method), 
                transforms.CenterCrop(self.image_size),
                _transform
            ])
            if task == 'segment_semantic':
                _new_transform = transforms.Compose([
                    _new_transform,
                    semseg_remap_inplace
                ])
            self.transform[task] = _new_transform
