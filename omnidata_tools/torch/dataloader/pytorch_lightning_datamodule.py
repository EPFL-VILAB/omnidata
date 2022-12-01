from .component_datasets import *

from fastcore.basics import store_attr, ifnone
from pytorch_lightning import seed_everything
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler, WeightedRandomSampler
from typing import Optional, Dict
import logging, loguru

class OmnidataDataModule(LightningDataModule):
    def __init__(self,
                 tasks = ('rgb', 'normal', 'mask_valid'),
                 train_datasets_to_options: Optional[Dict] = None,
                 eval_datasets_to_options:  Optional[Dict] = None,
                 shared_options:            Optional[Dict] = None,
                 train_options:             Optional[Dict] = None,
                 eval_options:              Optional[Dict] = None,
                 dataloader_kwargs      = None,
                 logger: logging.Logger = None
                ):
        super().__init__()
        store_attr()
        self.logger = ifnone(logger, loguru.logger)
        if 'batch_size' not in self.dataloader_kwargs: self.dataloader_kwargs['batch_size'] = 1
        self.drop_last = dataloader_kwargs.pop('drop_last', False)
        self._setup_datasets()

    @staticmethod
    def load_config(config_file):
        with open(config_file, 'r') as stream:
            config = yaml.safe_load(stream)
        return config

    def _setup_datasets(self, stage=None):
        # Some options are unique to the train set
        self.trainsets = {}
        for dataset_name, dataset_opts in self.train_datasets_to_options.items():
            opts_dict = {
                'split':'train',
                'tasks': self.tasks,
                **self.shared_options,
                **dataset_opts,
                **self.train_options
            }
            self.trainsets[dataset_name] =  eval(dataset_name)(eval(dataset_name).Options(**opts_dict))
            self.logger.info(f'Train set ({dataset_name}) contains {len(self.trainsets[dataset_name])} samples.')
        self.logger.success('Finished loading training sets.')
        
        # Sompe options are unique to the val set
        self.val_dataset_names, self.valsets = [], []
        for dataset_name, dataset_opts in self.eval_datasets_to_options.items():
            opts_dict = {
                'split':'val',
                'tasks': self.tasks,
                **self.shared_options,
                **dataset_opts,
                **self.eval_options,
            }
            self.valsets.append(eval(dataset_name)(eval(dataset_name).Options(**opts_dict)))
            self.val_dataset_names.append(dataset_name)
            self.logger.info(f'Val set ({dataset_name}) contains {len(self.valsets[-1])} samples.')
        self.logger.success('Loaded validation sets.')


    def train_dataloader(self):
        # Train dataloader ensures each of the k datasets makes up 1/k of the batch
        trainsets = self.trainsets.values()
        trainsets_counts = [len(trainset) for trainset in trainsets]

        dataset_sample_count = torch.tensor(trainsets_counts)
        weights = 1. / dataset_sample_count.float()
        samples_weight = []
        for w, count in zip(weights, dataset_sample_count):
            samples_weight += [w] * count
        samples_weight = torch.tensor(samples_weight)
        
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        train_dataset = ConcatDataset(trainsets)
        return DataLoader(
            train_dataset,
            sampler=sampler,
            drop_last=self.drop_last,
            **self.dataloader_kwargs
        )

    def val_dataloader(self):
        return [
            DataLoader(
                valset,
                drop_last=False,
                shuffle=True,
                **self.dataloader_kwargs
            )
            for valset in self.valsets
        ]