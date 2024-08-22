import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.classification import BinaryF1Score

from src.utils import get_intervals, get_confusion_matrix, draw_confusion_matrix, vector2prediction

def ddp_setup(world_size, rank):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def ddp_destroy():
    dist.destroy_process_group()

class TrainAgent:
    def __init__(self, device, model, optimizer, criterion, scheduler, logger=None):
        self.local_rank = device
        self.global_rank = int(os.environ["SLURM_PROCID"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        
        if self.world_size > 1:
            process_group = dist.new_group()
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group).to(self.local_rank)
            self.model = DDP(self.model, find_unused_parameters=True, device_ids=[self.local_rank])
        else:
            self.model = model.to(self.local_rank)


        self.n_classes = self.model.classifier.out_features
        if self.global_rank == 0:
            self._init_enviroment(logger)
            
    def train(self, n_epochs, train_loader, valid_loader) -> None:
        for epoch in range(1, n_epochs):
            ## Train
            self.stage = 'train'
            self.model.train()

            self._init_records()
            self._run_epoch(train_loader)
            self._reduce_records()
            self._process_records()
            self._log_scalars()


            ## Validation
            self.stage = 'valid'
            self.model.eval()

            with torch.no_grad():
                self._init_records()
                self._run_epoch(valid_loader)
                self._reduce_records()
                self._process_records()
                self._log_scalars()


            ## Creating snapshots
            if self.global_rank == 0:
                self._save_snapshot("LATEST.PTH")
                
                cur_metric = self.records['RMSE'].item()
                if cur_metric < self.best_metric:
                    self._save_snapshot("BEST.PTH")
                    self.best_metric = cur_metric

                    self.logger['metrics/best_metric'] = self.best_metric
                    self.logger['metrics/best_epoch'] = self.current_epoch

                self.current_epoch += 1

    def _run_epoch(self, loader) -> None:
        for i, (_, features, targets) in enumerate(loader):
            
            features = features.to(self.local_rank)
            targets = targets.to(self.local_rank)

            output = self.model(features)
            loss = self.criterion(output, targets)
    
            if self.stage == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            targets = targets.detach()
            output = output.detach()
            probabilities = nn.Softmax(dim=1)(output)
            predictions = torch.argmax(probabilities, dim=1)

            record = self._compute_metrics(predictions, targets)
            record['loss'] = loss.item()

            batch_size = targets.shape[0]
            confusion_matrix = get_confusion_matrix(targets, predictions, self.n_classes)
            self._update_records(record, batch_size, confusion_matrix)
        
        if not self.scheduler is None and self.stage == 'train':
            self.scheduler.step()

    def _compute_metrics(self, prediction, target):
        record = {}
        record.update({'RMSE' : torch.pow(10*(target - prediction), 2).sum()})
        
        return record

    def _init_records(self) -> None:
        self.records = {}
        self.records.update({'RMSE' : torch.tensor(0.0, device=self.local_rank)})
        self.records.update({'loss' : torch.tensor(0.0, device=self.local_rank)})

        self.n_samples = torch.tensor(0, device=self.local_rank)
        self.confusion_matrix = torch.zeros(self.n_classes, self.n_classes, device=self.local_rank)

    def _update_records(self, record, n_samples, confusion_matrix) -> None:
        for key in self.records.keys():
            self.records[key] += record[key]

        self.n_samples += n_samples
        self.confusion_matrix += confusion_matrix

    def _reduce_records(self) -> None:
        if self.world_size > 1:
            for key in self.records.keys():
                dist.reduce(self.records[key], dst=0)
            
            dist.reduce(self.n_samples, dst=0)
            dist.reduce(self.confusion_matrix, dst=0)

    def _process_records(self):
        if self.global_rank == 0:
            self.confusion_matrix = draw_confusion_matrix(self.confusion_matrix.cpu().numpy())
                
            for key in self.records.keys():
                self.records[key] = torch.nan_to_num(self.records[key] / self.n_samples, 0)
                
            self.records['RMSE'] = torch.sqrt(self.records['RMSE'])
        
    def _log_scalars(self) -> None:
        
        if self.global_rank == 0:
            
            for key in self.records.keys():
                self.logger[f'metrics/{self.stage}/{key}'].append(self.records[key].item())
            
            self.logger[f'metrics/{self.stage}/cm'].append(self.confusion_matrix)
                
            if self.stage == 'valid':
                self.logger['metrics/lr'].append(self.optimizer.param_groups[0]['lr'])

    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path, map_location=f"cuda:{self.local_rank}")

        if self.world_size > 1:
            self.model.module.load_state_dict(snapshot["PARAMS"])
        else: 
            self.model.load_state_dict(snapshot["PARAMS"])
        if not self.scheduler is None and not snapshot.get("SCHEDULER") is None:
            self.scheduler.load_state_dict(snapshot["SCHEDULER"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER"])
        self.current_epoch = snapshot["CURRENT_EPOCH"]

        print(f"Checkpoint loaded from {snapshot_path}")

    def _save_snapshot(self, snapshot_name):
        snapshot_path = os.path.join(self.snapshots_root, snapshot_name)
        snapshot = dict()
        
        snapshot["PARAMS"] = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
        snapshot["OPTIMIZER"] = self.optimizer.state_dict()
        snapshot["CURRENT_EPOCH"] = self.current_epoch
        snapshot["SCHEDULER"] = self.scheduler.state_dict() if not self.scheduler is None else None

        torch.save(snapshot, snapshot_path)
        print(f"Epoch {self.current_epoch} | Training snapshot saved at {snapshot_path}")

    def _init_enviroment(self, logger):
        self.logger = logger

        self.snapshots_root = os.path.join('exp', self.logger['sys/id'].fetch())
        if not os.path.exists('exp'):
            os.mkdir('exp')
        if not os.path.exists(self.snapshots_root):
            os.mkdir(self.snapshots_root)

        # Init counters

        self.current_epoch = 1
        self.best_metric = float("inf")
        
class BayesianTrainAgent(TrainAgent):
    def __init__(self, device, model, optimizer, criterion, scheduler, logger=None):
        super(BayesianTrainAgent, self).__init__(device, model, optimizer, criterion, scheduler, logger)

        self.n_classes = self.criterion.M

    def _run_epoch(self, loader) -> None:
        for i, (_, features, target) in enumerate(loader):
            
            features = features.to(self.local_rank)
            targets = target.to(self.local_rank)
            targets = torch.squeeze(targets, dim=1)

            outputs = self.model(features)
            outputs = torch.squeeze(outputs, dim=1)
            loss = self.criterion(outputs, targets)
            # print(loss, self.criterion.cutpoints)
    
            if self.stage == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            targets = targets.detach()
            outputs = outputs.detach()
            predictions = get_intervals(outputs, self.criterion.cutpoints)

            record = self._compute_metrics(predictions, targets)
            record['loss'] = loss.item()

            batch_size = targets.shape[0]
            confusion_matrix = get_confusion_matrix(targets, predictions, self.n_classes)
            self._update_records(record, batch_size, confusion_matrix)
        
        if not self.scheduler is None and self.stage == 'train':
            self.scheduler.step() 

    def _log_scalars(self) -> None:
    
        if self.global_rank == 0:
            
            for key in self.records.keys():
                self.logger[f'metrics/{self.stage}/{key}'].append(self.records[key].item())
            
            self.logger[f'metrics/{self.stage}/cm'].append(self.confusion_matrix)
                
            if self.stage == 'valid':
                self.logger['metrics/lr'].append(self.optimizer.param_groups[0]['lr'])
                self.logger['model/cutpoints'] = " ".join([str(e) for e in self.criterion.cutpoints.tolist()])

    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path, map_location=f"cuda:{self.local_rank}")

        if self.world_size > 1:
            self.model.module.load_state_dict(snapshot["PARAMS"])
        else: 
            self.model.load_state_dict(snapshot["PARAMS"])
        if not self.scheduler is None and not snapshot.get("SCHEDULER") is None:
            self.scheduler.load_state_dict(snapshot["SCHEDULER"])
        self.criterion.load_state_dict(snapshot["CUTPOINTS"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER"])
        self.current_epoch = snapshot["CURRENT_EPOCH"]

        print(f"Checkpoint loaded from {snapshot_path}")

    def _save_snapshot(self, snapshot_name):
        snapshot_path = os.path.join(self.snapshots_root, snapshot_name)
        snapshot = dict()
        
        snapshot["PARAMS"] = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
        snapshot["CUTPOINTS"] = self.criterion.cutpoints
        snapshot["OPTIMIZER"] = self.optimizer.state_dict()
        snapshot["CURRENT_EPOCH"] = self.current_epoch
        snapshot["SCHEDULER"] = self.scheduler.state_dict() if not self.scheduler is None else None

        torch.save(snapshot, snapshot_path)
        print(f"Epoch {self.current_epoch} | Training snapshot saved at {snapshot_path}")

class OrdinalBinTrainAgent(TrainAgent):
    def __init__(self, device, model, optimizer, criterion, scheduler, logger=None):
        super(OrdinalBinTrainAgent, self).__init__(device, model, optimizer, criterion, scheduler, logger)
        self.n_classes += 1

    def train(self, n_epochs, train_loader, valid_loader) -> None:
        for epoch in range(1, n_epochs):
            ## Train
            self.stage = 'train'
            self.model.train()

            self._init_records()
            self._run_epoch(train_loader)
            self._reduce_records()
            self._process_records()
            self._log_scalars()


            ## Validation
            self.stage = 'valid'
            self.model.eval()

            with torch.no_grad():
                self._init_records()
                self._run_epoch(valid_loader)
                self._reduce_records()
                self._process_records()
                self._log_scalars()
                self._estimate_threshold()


            ## Creating snapshots
            if self.global_rank == 0:
                self._save_snapshot("LATEST.PTH")
                
                cur_metric = self.records['RMSE'].item()
                if cur_metric < self.best_metric:
                    self._save_snapshot("BEST.PTH")
                    self.best_metric = cur_metric
                    
                    self.logger['metrics/best_threshold'] = self.threshold
                    self.logger['metrics/best_metric'] = self.best_metric
                    self.logger['metrics/best_epoch'] = self.current_epoch

                self.current_epoch += 1

    def _run_epoch(self, loader) -> None:
        for i, (_, features, target) in enumerate(loader):
            
            features = features.to(self.local_rank)
            targets = target.to(self.local_rank)

            output = self.model(features)
            loss = self.criterion(output, targets)
    
            if self.stage == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            targets = targets.detach()
            targets = vector2prediction(targets)
            
            output = output.detach()
            probabilities = nn.Sigmoid()(output)
            predictions = probabilities > self.threshold
            predictions = predictions.to(torch.int64)
            predictions = vector2prediction(predictions)

            record = self._compute_metrics(predictions, targets)
            record['loss'] = loss.item()

            confusion_matrix = get_confusion_matrix(targets, predictions, self.n_classes)
            self._update_records(record, targets, probabilities, confusion_matrix)
        
        if not self.scheduler is None and self.stage == 'train':
            self.scheduler.step()

    def _init_records(self) -> None:
        self.records = {}
        self.records.update({'RMSE' : torch.tensor(0.0, device=self.local_rank)})
        self.records.update({'loss' : torch.tensor(0.0, device=self.local_rank)})

        self.n_samples = torch.tensor(0, device=self.local_rank)
        self.confusion_matrix = torch.zeros(self.n_classes, self.n_classes, device=self.local_rank)
        
        if self.stage == 'valid':
            self.probabilities = torch.empty((0, self.n_classes - 1), device=self.local_rank)
            self.targets = torch.empty(0, device=self.local_rank)
    
    def _update_records(self, record, targets, probabilities, confusion_matrix):
        for key in self.records.keys():
            self.records[key] += record[key]

        self.n_samples += targets.shape[0]
        self.confusion_matrix += confusion_matrix
        if self.stage == 'valid':
            self.targets = torch.hstack([self.targets, targets])
            self.probabilities = torch.vstack([self.probabilities, probabilities])
            
    def _reduce_records(self) -> None:
        if self.world_size > 1:
            for key in self.records.keys():
                dist.reduce(self.records[key], dst=0)
            
            dist.reduce(self.n_samples, dst=0)
            dist.reduce(self.confusion_matrix, dst=0)
            
        if self.stage == 'valid' and self.world_size > 1:
            
            if self.global_rank == 0:
                
                targets_acc = [torch.zeros_like(self.targets, device=self.local_rank) for _ in range(self.world_size)] 
                probabilities_acc = [torch.zeros_like(self.probabilities, device=self.local_rank) for _ in range(self.world_size)] 
                
                dist.gather(self.targets, targets_acc, dst=0)
                dist.gather(self.probabilities, probabilities_acc, dst=0)
                
                self.targets = torch.hstack([targets_acc[i] for i in range(self.world_size)])
                self.probabilities = torch.hstack([probabilities_acc[i] for i in range(self.world_size)])
                
            else:

                dist.gather(self.targets, dst=0)    
                dist.gather(self.probabilities, dst=0)           

    def _process_records(self):
        if self.global_rank == 0:
            self.confusion_matrix = draw_confusion_matrix(self.confusion_matrix.cpu().numpy())
                
            for key in self.records.keys():
                self.records[key] = torch.nan_to_num(self.records[key] / self.n_samples, 0)
                
            self.records['RMSE'] = torch.sqrt(self.records['RMSE'])
            
    def _estimate_threshold(self):
        
        if self.global_rank == 0:
        
            best_rmse = float("inf")

            for threshold in self.threshold_range:
                predictions = self.probabilities > threshold
                predictions = predictions.to(torch.int64)
                predictions = vector2prediction(predictions)

                rmse = torch.pow(10*(self.targets - predictions), 2).mean()
                rmse = torch.sqrt(rmse)

                if rmse < best_rmse:
                    self.threshold = threshold
                    best_rmse = rmse
                
        elif self.world_size > 1:
            dist.broadcast(self.threshold, src=0)  

    def _log_scalars(self) -> None:
        
        if self.global_rank == 0:
        
            for key in self.records.keys():
                self.logger[f'metrics/{self.stage}/{key}'].append(self.records[key].item())

            self.logger[f'metrics/{self.stage}/cm'].append(self.confusion_matrix)
                
            if self.stage == 'valid':
                self.logger['metrics/lr'].append(self.optimizer.param_groups[0]['lr'])
                self.logger['metrics/threshold'].append(self.threshold)
 
    def _init_enviroment(self, logger):
        self.logger = logger

        self.snapshots_root = os.path.join('exp', self.logger['sys/id'].fetch())
        if not os.path.exists('exp'):
            os.mkdir('exp')
        if not os.path.exists(self.snapshots_root):
            os.mkdir(self.snapshots_root)

        # Init counters

        self.current_epoch = 1

        self.threshold = torch.tensor(0.5, device=self.local_rank)
        self.threshold_range = torch.arange(0.25, 0.75, 0.05, device=self.local_rank)

        self.best_metric = float("inf")
        self.best_threshold = torch.tensor(0.5, device=self.local_rank)

class BinaryTrainAgent(OrdinalBinTrainAgent):
    def __init__(self, device, model, optimizer, criterion, scheduler, logger=None):
        super(BinaryTrainAgent, self).__init__(device, model, optimizer, criterion, scheduler, logger) 
        
        self.f1_score = BinaryF1Score().to(self.local_rank)
        if self.global_rank == 0:
            self.best_metric = 0
        
    def train(self, n_epochs, train_loader, valid_loader) -> None:
        for epoch in range(1, n_epochs):
            ## Train
            self.stage = 'train'
            self.model.train()

            self._init_records()
            self._run_epoch(train_loader)
            self._reduce_records()
            self._process_records()
            self._log_scalars()


            ## Validation
            self.stage = 'valid'
            self.model.eval()

            with torch.no_grad():
                self._init_records()
                self._run_epoch(valid_loader)
                self._reduce_records()
                self._process_records()
                self._log_scalars()
                self._estimate_threshold()


            ## Creating snapshots
            if self.global_rank == 0:
                self._save_snapshot("LATEST.PTH")
                
                cur_metric = self.records['F1-score'].item()
                if cur_metric > self.best_metric:
                    self._save_snapshot("BEST.PTH")
                    self.best_metric = cur_metric

                    self.logger['metrics/best_threshold'] = self.threshold
                    self.logger['metrics/best_metric'] = self.best_metric
                    self.logger['metrics/best_epoch'] = self.current_epoch

                self.current_epoch += 1

    def _run_epoch(self, loader) -> None:
        for i, (_, features, targets) in enumerate(loader):
            
            features = features.to(self.local_rank)
            targets = targets.to(self.local_rank)

            output = self.model(features)
            loss = self.criterion(output, targets)
    
            if self.stage == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            targets = targets.detach().flatten()
            targets = targets.to(torch.int64)
            output = output.detach().flatten()
            probabilities = nn.Sigmoid()(output)
            predictions = probabilities > self.threshold
            predictions = predictions.to(torch.int64)

            record = {'loss' : loss.item()}

            confusion_matrix = get_confusion_matrix(targets, predictions, self.n_classes)
            self._update_records(record, targets, probabilities, confusion_matrix)
        
        if not self.scheduler is None and self.stage == 'train':
            self.scheduler.step()

    def _init_records(self) -> None:
        self.records = {}
        self.records.update({'loss' : torch.tensor(0.0, device=self.local_rank)})

        self.n_samples = torch.tensor(0, device=self.local_rank)
        self.confusion_matrix = torch.zeros(self.n_classes, self.n_classes, device=self.local_rank)

        self.probabilities = torch.empty(0, device=self.local_rank)
        self.targets = torch.empty(0, device=self.local_rank)

    def _update_records(self, record, targets, probabilities, confusion_matrix):
        for key in self.records.keys():
            self.records[key] += record[key]

        self.n_samples += targets.shape[0]
        self.confusion_matrix += confusion_matrix
        
        self.targets = torch.hstack([self.targets, targets])
        self.probabilities = torch.hstack([self.probabilities, probabilities])
        
    def _process_records(self) -> None:            
        if self.global_rank == 0:
            self.confusion_matrix = draw_confusion_matrix(self.confusion_matrix.cpu().numpy())  
                
            for key in self.records.keys():
                self.records[key] = torch.nan_to_num(self.records[key] / self.n_samples, 0)
                
            predictions = self.probabilities > self.threshold
            predictions = predictions.to(torch.int64)
            self.records.update({'F1-score' : self.f1_score(predictions, self.targets)})

    def _estimate_threshold(self):
        
        if self.global_rank == 0:
        
            best_f1 = 0

            for threshold in self.threshold_range:
                predictions = self.probabilities > threshold
                predictions = predictions.to(torch.int64)

                f1 = self.f1_score(predictions, self.targets)

                if f1 > best_f1:
                    self.threshold = threshold
                    best_f1 = f1
                
        elif self.world_size > 1:
            dist.broadcast(self.threshold, src=0)  
       
class EmbedingsAgent():
    def __init__(self, model, device):
        self.device = device
        self.model = model
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def get_embedings(self, X):

        X = X.to(self.device)
        
        with torch.no_grad():
            output = self.model(X)
            
        return output.cpu().numpy()
    
    def load_weights(self, path):
        weights = torch.load(path, map_location=self.device)
        self.model.load_state_dict(weights['PARAMS'])
        print(f"Weights are loaded from {path}")
        
    def truncate_model(self):
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
class InferenceAgent():
    def __init__(self, model, device):
        self.device = device
        self.model = model
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def predict(self, X):

        X = X.to(self.device)
        
        with torch.no_grad():
            output = self.model(X)
            
        return output
    
    def load_weights(self, path):
        weights = torch.load(path, map_location=self.device)
        self.model.load_state_dict(weights['PARAMS'])
        print(f"Weights are loaded from {path}")