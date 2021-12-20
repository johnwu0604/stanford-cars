import os
import numpy as np
import torch
import mlflow
import torch.nn as nn

from argparse import ArgumentParser
from torch.nn import functional as F
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from torchmetrics.functional.classification import accuracy
from pytorch_lightning.loggers import MLFlowLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim import Adam
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from typing import Optional
from torch.nn import Module

BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

def _make_trainable(module: Module) -> None:
    """Unfreezes a given module.
    Args:
        module: The module to unfreeze
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def _recursive_freeze(module: Module,
                      train_bn: bool = True) -> None:
    """Freezes the layers of a given module.
    Args:
        module: The module to freeze
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            # Make the BN layers trainable
            _make_trainable(module)
    else:
        for child in children:
            _recursive_freeze(module=child, train_bn=train_bn)


def freeze(module: Module,
           n: Optional[int] = None,
           train_bn: bool = True) -> None:
    """Freezes the layers up to index n (if n is not None).
    Args:
        module: The module to freeze (at least partially)
        n: Max depth at which we stop freezing the layers. If None, all
            the layers of the given module will be frozen.
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    n_max = len(children) if n is None else int(n)

    for child in children[:n_max]:
        _recursive_freeze(module=child, train_bn=train_bn)

    for child in children[n_max:]:
        _make_trainable(module=child)

class ResNet50(LightningModule):

    def __init__(self, 
                data_dir: str,
                train_bn: bool = True,
                batch_size: int = 16,
                lr: float = 1e-3,
                num_workers: int = 4,
                hidden_1: int = 1024,
                hidden_2: int = 512,
                epoch_freeze: int = 8,
                total_steps: int = 15,
                pct_start: float = 0.2,
                anneal_strategy: str = 'cos',
                **kwargs):
        super().__init__()
        self.train_bn = train_bn
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.epoch_freeze = epoch_freeze
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.__build_model()
        
    def __build_model(self):
        num_target_classes = 196
        backbone = models.resnet50(pretrained=True)
    
        _layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*_layers)

        _fc_layers = [nn.Linear(2048, self.hidden_1),
                     nn.Linear(self.hidden_1, self.hidden_2),
                     nn.Linear(self.hidden_2, num_target_classes)]
        self.fc = nn.Sequential(*_fc_layers)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x
    
    def train(self, mode=True):
        super().train(mode=mode)
        epoch = self.current_epoch
        if epoch < self.epoch_freeze and mode:
            freeze(module=self.feature_extractor,
                   train_bn=self.train_bn) 
            
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_logits = self.forward(x)
        train_loss = F.cross_entropy(y_logits, y)
        acc = accuracy(y_logits, y)
        self.log('train_loss', train_loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_logits = self.forward(x)
        val_loss = F.cross_entropy(y_logits, y)
        acc = accuracy(y_logits, y)
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        if self.current_epoch < self.epoch_freeze:
            optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
            return optimizer
        else:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)     
            scheduler = OneCycleLR(optimizer,
                            max_lr=self.lr,
                            total_steps=self.total_steps,
                            pct_start=self.pct_start, anneal_strategy=self.anneal_strategy)
        return [optimizer], [scheduler]

    def setup(self, stage: str):

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        train_transforms = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.RandomCrop(350),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std, inplace=True)
        ])
        train = ImageFolder(self.data_dir + '/train', train_transforms)

        val_transforms = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std, inplace=True)
        ])
        val = ImageFolder(self.data_dir + '/test', val_transforms)
        valid, _ = random_split(val, [len(val), 0])

        self.train_dataset = train
        self.val_dataset = valid

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            shuffle=True,
                            pin_memory=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            shuffle=False,
                            pin_memory=True)

def main(args):

    seed_everything(42)
    model = ResNet50(data_dir=args.data_dir, batch_size=args.batch_size)

    mlf_logger = MLFlowLogger()
    
    checkpoint_cb = ModelCheckpoint(dirpath='./cars', filename ='cars-{epoch:02d}-{val_acc:.4f}', monitor='val_acc', mode='max')
    early_stop_cb = EarlyStopping(patience=5, monitor='val_acc', mode='max')

    trainer = Trainer(
        gpus=args.gpus,
        logger=mlf_logger,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        deterministic=True,
        num_sanity_val_steps=args.num_sanity_val_steps,
        val_check_interval=args.val_check_interval,
        callbacks=[checkpoint_cb, early_stop_cb]
    )

    trainer.fit(model)
    mlflow.pytorch.save_model(model, args.output_dir)

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--gpus", default=0, type=int)
    parser.add_argument("--max_epochs", default=25, type=int)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_sanity_val_steps", default=1, type=int)
    parser.add_argument("--val_check_interval", default=1.0, type=float)
    parser.add_argument("--output_dir", default="model", type=str)
    return parser.parse_args()

if __name__ == '__main__':
    main(get_args())