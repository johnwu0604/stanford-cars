import mlflow
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torchmetrics.functional.classification import accuracy
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import Adam
from torchvision import models
from typing import Optional
from torch.nn import Module

BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

def _make_trainable(module: Module) -> None:
    '''Unfreezes a given module.
    Args:
        module: The module to unfreeze
    '''
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def _recursive_freeze(module: Module,
                      train_bn: bool = True) -> None:
    '''Freezes the layers of a given module.
    Args:
        module: The module to freeze
        train_bn: If True, leave the BatchNorm layers in training mode
    '''
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
    '''Freezes the layers up to index n (if n is not None).
    Args:
        module: The module to freeze (at least partially)
        n: Max depth at which we stop freezing the layers. If None, all
            the layers of the given module will be frozen.
        train_bn: If True, leave the BatchNorm layers in training mode
    '''
    children = list(module.children())
    n_max = len(children) if n is None else int(n)

    for child in children[:n_max]:
        _recursive_freeze(module=child, train_bn=train_bn)

    for child in children[n_max:]:
        _make_trainable(module=child)

class ResNet152(pl.LightningModule):

    def __init__(self, 
                train_bn: bool = True,
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
        self.lr = lr
        self.num_workers = num_workers
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.epoch_freeze = epoch_freeze
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.save_hyperparameters()
        self.__build_model()
        
    def __build_model(self):
        num_target_classes = 196
        backbone = models.resnet152(pretrained=True)
    
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
        self.log('acc', acc, prog_bar=True)
        return train_loss

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
            optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)     
            scheduler = OneCycleLR(optimizer,
                            max_lr=self.lr,
                            total_steps=self.total_steps,
                            pct_start=self.pct_start, anneal_strategy=self.anneal_strategy)
        return [optimizer], [scheduler]