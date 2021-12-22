import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

class CarData(pl.LightningDataModule):

    def __init__(self, batch_size, data_dir):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir

    def setup(self, stage):
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
        self.train_dataset = train

        val_transforms = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std, inplace=True)
        ])
        val = ImageFolder(self.data_dir + '/test', val_transforms)
        self.val_dataset = val

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=4, 
            shuffle=True,
            pin_memory=True)

    def val_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=4, 
            shuffle=False,
            pin_memory=True)

    def get_class_idx(self):
        return self.train_dataset.class_to_idx 
