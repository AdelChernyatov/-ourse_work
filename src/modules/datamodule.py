from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import pandas as pd
from modules.dataset import CustomDatasetWithContours


class UnetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, config=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.Resize((576, 576)), transforms.ToTensor()])
        self.batch_size = config['batch_size']

    def setup(self, stage: str, contours=False):
        if stage == "fit" or stage is None:
            train_df = pd.read_csv(self.data_dir + 'train_df.csv')
            self.train_dataset = CustomDatasetWithContours(train_df)
            validation_df = pd.read_csv(self.data_dir + 'val_df.csv')
            self.validation_dataset = CustomDatasetWithContours(validation_df)
        if stage == "test":
            test_df = pd.read_csv(self.data_dir + 'test_df.csv')
            self.test_dataset = CustomDatasetWithContours(test_df)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)