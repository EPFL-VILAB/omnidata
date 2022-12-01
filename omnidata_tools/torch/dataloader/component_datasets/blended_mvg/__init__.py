from   typing import Optional, List, Callable, Union, Dict, Any
import logging
import os
import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map
import multiprocessing as mp
# from   PIL import Image
from   torchvision import transforms
from ...segment_instance import random_colors
from ...splits import get_splits
from ...omnidata_dataset import OmnidataDataset, load_subfolder


def filter_blendedMVS():
    data = pd.read_csv('/scratch/roman/Omnidata/loss_stats/blendedMVS_sorted.csv', index_col=0)
    thresholds = [1000, 10000, 20000]
    filtered_bpv = {}
    for thresh in thresholds:
        count = 0
        d = defaultdict(list)
        for index, row in data.iterrows():
            if count > thresh: break
            d[row['building']].append(int(row['point']))
            count += 1
        filtered_bpv[thresh] = d
    return filtered_bpv

blended_mvg_bad_scenes = ['5bf21799d43923194842c001',
                        '000000000000000000000001',
                        '000000000000000000000000',
                        '5b21e18c58e2823a67a10dd8',
                        '5ab8b8e029f5351f7f2ccf59',
                        '00000000000000000000000c',
                        '00000000000000000000000b',
                        '58cf4771d0f5fb221defe6da',
                        '5c2b3ed5e611832e8aed46bf',
                        '584a7333fe3cb463906c9fe6',
                        '585ee0632a57cc11d4933608',
                        '5983012d1bd4b175e70c985a',
                        '58a0dd1a3d0b4542479a28f3',
                        '58a44463156b87103d3ed45e']



###############################################################################
# Make dataset 
###############################################################################

    
class BlendedMVGDataset(OmnidataDataset):
    def __init__(self, options: OmnidataDataset.Options, logger: logging.Logger=None):
        self.dataset_name = 'blended_mvg'
        super().__init__(options, logger)
        for task, _transform in self.transform.items():
            resize_method = Image.BILINEAR if task in ['rgb'] else Image.NEAREST
            _new_transform = transforms.Compose([
                transforms.Resize(self.image_size, resize_method), 
                transforms.CenterCrop(self.image_size),
                _transform
            ])
            if task == 'semantic':
                _new_transform = transforms.Compose([
                    _new_transform,
                    semseg_remap_inplace
                ])
            self.transform[task] = _new_transform

    def _folder_in_split(self, folder, split):
        row = self.split_df.loc[ self.split_df['id']==folder]
        if row.empty or row.iloc[0]['id'] in blended_mvg_bad_scenes: return False
        return (not row.empty and row.iloc[0][split] == 1)
