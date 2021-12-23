import os
import mlflow
import torch
import json
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from model import ResNet152
from data import CarData

def main(args):

    seed_everything(42)
    mlflow.pytorch.autolog()
    
    model = ResNet152()
    data = CarData(args.batch_size, args.data_dir)

    checkpoint_cb = ModelCheckpoint(dirpath='./cars', filename ='cars-{epoch:02d}-{val_acc:.4f}', monitor='val_acc', mode='max')
    early_stop_cb = EarlyStopping(patience=5, monitor='val_acc', mode='max')
    trainer = Trainer(
        gpus=args.gpus,
        strategy='ddp',
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        deterministic=True,
        num_sanity_val_steps=args.num_sanity_val_steps,
        val_check_interval=args.val_check_interval,
        callbacks=[checkpoint_cb, early_stop_cb]
    )
    trainer.fit(model, data)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    torch.save(model.state_dict(), args.output_dir + '/model.pth')
    with open(args.output_dir + '/classes.json', 'w') as f:
        json.dump(data.get_class_idx(), f)
        
def get_args():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--gpus', default=0, type=int)
    parser.add_argument('--max_epochs', default=25, type=int)
    parser.add_argument('--max_steps', default=-1, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_sanity_val_steps', default=1, type=int)
    parser.add_argument('--val_check_interval', default=1.0, type=float)
    parser.add_argument('--output_dir', default='outputs', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    main(get_args())