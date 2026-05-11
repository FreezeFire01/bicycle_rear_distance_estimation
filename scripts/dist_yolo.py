"""
Modifikácie ktoré sme spravili v ultralytics:
    - head.py: pridaná DetectDist trieda
    - loss.py: pridaná v8DistDetectionLoss
    - dataset.py: pridaná YOLODistDataset
    - utils.py (data): verify_image_label akceptuje 6 stĺpcov
    - tasks.py: DetectDist v import sekcii a parse_model frozenset

Tu doplňujeme:
    - DistDetectionModel — DetectionModel s init_criterion → v8DistDetectionLoss
    - DistDetectionTrainer — používa YOLODistDataset
    - DistDetectionValidator — distance-aware metrics (placeholder)
    - DistYOLO — high-level API ako YOLO trieda
"""

import os
from pathlib import Path
from copy import copy
import torch
import numpy as np

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.engine.validator import BaseValidator
from ultralytics.engine.model import Model
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DistDetectionLoss
from ultralytics.data.dataset import YOLODistDataset
from ultralytics.data.build import build_dataloader
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model
from ultralytics.utils import LOGGER, RANK



class DistDetectionModel(DetectionModel):
    """DetectionModel s init_criterion vracia v8DistDetectionLoss."""

    def init_criterion(self):
        """Inicializuj distance-aware loss."""
        return v8DistDetectionLoss(self)

def build_dist_dataset(cfg, img_path, batch, data, mode='train', rect=False, stride=32):
    """Build YOLODistDataset (analogicky k build_yolo_dataset, ale s distance support)."""
    return YOLODistDataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == 'train',
        hyp=cfg,
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == 'train' else 0.5,
        prefix=f'{mode}: ',
        task='detect',
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == 'train' else 1.0,
    )


class DistDetectionTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_dist_dataset(
            self.args, img_path, batch, self.data,
            mode=mode, rect=mode == 'val', stride=gs
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DistDetectionModel(
            cfg,
            nc=self.data['nc'],
            ch=self.data['channels'],
            verbose=verbose and RANK == -1
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss', 'dist_loss'
        return DistDetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix='train'):
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        return ('\n' + '%11s' * (4 + len(self.loss_names))) % (
            'Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size'
        )


class DistDetectionValidator(DetectionValidator):

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.distance_errors = []  # zhromažďuj distance errors per batch

    def init_metrics(self, model):
        """Inicializuj metrics + distance MAE tracking."""
        super().init_metrics(model)
        self.distance_errors = []

    def build_dataset(self, img_path, mode='val', batch=None):
        """Override: YOLODistDataset."""
        gs = 32
        if hasattr(self, 'model') and self.model is not None:
            try:
                gs = max(int(unwrap_model(self.model).stride.max()), 32)
            except (AttributeError, TypeError):
                gs = 32
        return build_dist_dataset(
            self.args, img_path, batch, self.data,
            mode=mode, rect=True, stride=gs
        )

    def get_stats(self):
        stats = super().get_stats()
        if self.distance_errors:
            mae = float(np.mean(self.distance_errors))
            # Convert from normalized to meters (x MAX_DISTANCE = 90m)
            mae_m = mae * 90.0
            stats['metrics/distance_mae_m'] = mae_m
            LOGGER.info(f"\n  Distance MAE: {mae_m:.2f}m (normalized: {mae:.4f})")
        return stats


class DistYOLO(Model):
    """
    Používanie:
        model = DistYOLO('dist-yolo11.yaml')
        results = model.train(data='dataset.yaml', epochs=100)
    """

    @property
    def task_map(self):
        """Map task → trainer/model/validator."""
        return {
            'detect': {
                'model': DistDetectionModel,
                'trainer': DistDetectionTrainer,
                'validator': DistDetectionValidator,
            }
        }