import pandas as pd
import os
import itertools
from socket import gethostname
import neptune
import copy
import json
import gc

import torch
import torch.nn as nn
from torch.distributed import is_initialized
import timm
import albumentations as A

from src.dataset import DamageDataset, OrdinalBinDamageDataset, BinaryDamageDataset
from src.agents import TrainAgent, OrdinalBinTrainAgent, BinaryTrainAgent, BayesianTrainAgent, ddp_setup, ddp_destroy
from src.utils import dict2str, estimate_maximum_batch_size, get_classes_priora_proba, get_costs_proportional
from src.losses import BayesianLoss

def main():
    world_size = int(os.environ["WORLD_SIZE"])
    global_rank = int(os.environ["SLURM_PROCID"])
    gpus_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])

    assert gpus_per_node <= torch.cuda.device_count()

    print(f"Hello from rank {global_rank} of {world_size} on {gethostname()} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)

    if world_size > 1:
        print("DDP is using")
        ddp_setup(world_size, global_rank)

        if global_rank == 0: 
            print(f"\nGroup initialized? {is_initialized()}", flush=True)
            
    else:
        print("DDP is not using")

    local_rank = global_rank - gpus_per_node * (global_rank // gpus_per_node)
    torch.cuda.set_device(local_rank)

    print(f"host: {gethostname()}, rank: {global_rank}, local_rank: {local_rank}")

    torch.cuda.empty_cache()
    gc.collect()


    parameters = {
        "batch_size" : 'auto',
        "n_epochs"   : 100,
        "base_lr"    : 0.0001,
        "train_csv"  : "train.csv",
        "valid_csv"  : "valid.csv",
        "distance"   : "torch.abs(x - y)"
    }

    model = {
        'name'       : 'efficientnet_b0',
        "loss"       : "BayesianLoss",
        "optimizer"  : 'Adam',
        "scheduler"  : None,
    }

    root = 'data/train'
    train_split = pd.read_csv(f'data/splits/{parameters["train_csv"]}', index_col='ID')
    valid_split = pd.read_csv(f'data/splits/{parameters["valid_csv"]}', index_col='ID')
    metadata = pd.read_csv('data/train_meta.csv', index_col='ID')

    distance = lambda x,y : torch.abs(x - y)
    n_classes = 11

    loss_parameters = {
        # "classes_prior_proba" : get_classes_priora_proba(train_split, metadata), 
        "classes_prior_proba" : torch.ones(n_classes, dtype=torch.float32) / n_classes
        "costs_proportional"  : get_costs_proportional(n_classes, distance),
        "kernel"              : torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(0.1507)).cdf,
        "n_classes"           : n_classes
    }
    
    net = timm.create_model(model['name'], pretrained=True, num_classes=1)
    criterion = BayesianLoss(**loss_parameters).to(local_rank)
    optimizer = torch.optim.Adam([
        {'params' : net.parameters(), "lr" : parameters['base_lr']}, 
        {'params' : criterion.parameters(), "lr" : parameters['base_lr']}
    ])
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2*parameters['base_lr'], total_steps=parameters['n_epochs'])
    scheduler = None


    if parameters['batch_size'] == 'auto':
        parameters['batch_size'] = estimate_maximum_batch_size(net, local_rank, (3, 512, 512))
    
    transforms = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5)
    ]
    
    train_loader, valid_loader = DamageDataset.get_dataloaders(root=root, 
                                                                train_split=train_split, 
                                                                valid_split=valid_split,
                                                                metadata=metadata, 
                                                                batch_size=parameters['batch_size'], 
                                                                transforms=transforms)
    
    if global_rank == 0:

        print("###")
        print("LOGGER INITIALIZATION")
        print("###")

        logger = neptune.init_run(
            project="GreekAI/ZINDI-cgiar",
            description='Bayesian classification, sigma^2 set to 0.1507', 
            source_files=["src/*.py", "train.py", "launch.sh"],
            api_token="",
        ) 

        fields = ['parameters', 'model', 'metrics', 'monitoring', 'images', 'snapshots', 'matric']
        for field in fields:
            try:
                del logger[field]
            except neptune.exceptions.MetadataInconsistency:
                pass

        logger['parameters'] = dict2str(copy.deepcopy(parameters))
        logger['model'] = dict2str(copy.deepcopy(model))
        logger['loss'] = dict2str(copy.deepcopy(loss_parameters))
    
        logger.wait()

    else:
        logger = None

    agent = BayesianTrainAgent(local_rank, net, optimizer, criterion, scheduler, logger)
    agent.train(parameters['n_epochs'], train_loader, valid_loader)
    
    if world_size > 1:
        ddp_destroy()

if __name__ == '__main__':
    main()
