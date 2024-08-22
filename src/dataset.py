import os
import numpy as np
import pandas as pd
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DamageDataset(Dataset):
    def __init__(self, root, dataframe, metadata, transforms=None):
        super(DamageDataset, self).__init__()
        
        self.root = root
        self.dataframe = dataframe 
        self.meta = metadata
        
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        height, width = (512,512)
        
        mandatory_transforms = [
            A.Resize(height=height, width=width, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=mean, std=std, always_apply=True),
            ToTensorV2()
        ]

        if transforms is None:
            self.transforms = A.Compose(mandatory_transforms) 
        else:
            self.transforms = A.Compose(transforms + mandatory_transforms)
            
        try:
            self.classes = np.unique(self.meta.loc[self.dataframe.index].extent)
            self.n_classes = self.classes.shape[0]
            self.class_shift = self.classes.min()
            
        except Exception as e:
            self.classes = None
            self.n_classes = 10
            self.class_shift = 0

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        # Features
        idx = self.dataframe.iloc[idx].name
        filename = self.meta.loc[idx].filename
                   
        features = cv2.imread(os.path.join(self.root,  filename))
        features = cv2.cvtColor(features, cv2.COLOR_BGR2RGB)
        features = np.nan_to_num(features, 0)
        features = features.astype(np.float32)
        
        features = self.transforms(image=features)['image']

        # Target
        try:
            target = self.meta.loc[idx].extent - self.class_shift
            target /= 10
            target = torch.tensor(target, dtype=torch.int64)

        except Exception as e:
            target = torch.tensor(0, dtype=torch.int64)
        target = torch.unsqueeze(target, dim=0)

        return idx, features, target

    @classmethod
    def get_dataloaders(cls, root, train_split, valid_split, metadata, batch_size, transforms=None):
        train_dataset = cls(root, train_split, metadata, transforms)
        valid_dataset = cls(root, valid_split, metadata, transforms)
        
        _get_dataloaders = cls.__get_dataloaders_ddp if int(os.environ["WORLD_SIZE"]) > 1 else cls.__get_dataloaders
        
        return _get_dataloaders(train_dataset, valid_dataset, batch_size)
    
    @staticmethod
    def __get_dataloaders(train_dataset, valid_dataset, batch_size):
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))

        valid_dataloader = DataLoader(valid_dataset,
                                      batch_size=batch_size,
                                      num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))
        
        return train_dataloader, valid_dataloader
    
    @staticmethod
    def __get_dataloaders_ddp(train_dataset, valid_dataset, batch_size):
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      drop_last=True,
                                      sampler=DistributedSampler(train_dataset,
                                                                 num_replicas=int(os.environ["WORLD_SIZE"]),
                                                                 rank=int(os.environ["SLURM_PROCID"])),
                                      num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))

        valid_dataloader = DataLoader(valid_dataset,
                                      batch_size=batch_size,
                                      sampler=DistributedSampler(valid_dataset,
                                                                 num_replicas=int(os.environ["WORLD_SIZE"]),
                                                                 rank=int(os.environ["SLURM_PROCID"])),
                                      num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))
        
        return train_dataloader, valid_dataloader

class OrdinalBinDamageDataset(DamageDataset):
    def __init__(self, root, dataframe, metadata, transforms=None):
        super(OrdinalBinDamageDataset, self).__init__(root, dataframe, metadata, transforms)

    def __getitem__(self, idx):
        # Features
        idx = self.dataframe.iloc[idx].name
        filename = self.meta.loc[idx].filename
                   
        features = cv2.imread(os.path.join(self.root,  filename))
        features = cv2.cvtColor(features, cv2.COLOR_BGR2RGB)
        features = np.nan_to_num(features, 0)
        features = features.astype(np.float32)
        
        features = self.transforms(image=features)['image']

        try:
            target = self.meta.loc[idx].extent - self.class_shift
            target //= 10
            bin_target = torch.zeros(self.n_classes - 1, dtype=torch.float32)
            bin_target[:target] = 1

        except Exception as e:
            bin_target = torch.zeros(self.n_classes - 1, dtype=torch.float32)

        return idx, features, bin_target
    
class BinaryDamageDataset(DamageDataset):
    def __init__(self, root, dataframe, metadata, transforms=None):
        super(BinaryDamageDataset, self).__init__(root, dataframe, metadata, transforms)
        
    def __getitem__(self, idx):
        # Features
        idx = self.dataframe.iloc[idx].name
        filename = self.meta.loc[idx].filename
                   
        features = cv2.imread(os.path.join(self.root,  filename))
        features = cv2.cvtColor(features, cv2.COLOR_BGR2RGB)
        features = np.nan_to_num(features, 0)
        features = features.astype(np.float32)
        
        features = self.transforms(image=features)['image']

        try:
            target = self.meta.loc[idx].extent
            target = 1 if target > 0 else 0
            target = torch.tensor(target, dtype=torch.float32)

        except Exception as e:
            target = torch.tensor(0, dtype=torch.float32)
        target = torch.unsqueeze(target, dim=0)

        return idx, features, target