import warnings
import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
from pathlib import Path
from clearml import Task
import pandas as pd
import yaml

from modules.model import UNet
from modules.dataset import CustomDatasetWithContours
from modules.datamodule import UnetDataModule

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, message="No audio backend is available.")


def main() -> None:
    logger.info('Start Unet training')

    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    pl.seed_everything(config['seed'], workers=True)

    task = Task.init(
        project_name=config['proj_name'],
        task_name=f'{config["exp_name"]}',
        auto_connect_frameworks=True,
    )

    dataframe_path = Path(config['dataframe_dir'])
    train_df = pd.read_csv(dataframe_path / 'train_df.csv')
    validation_df = pd.read_csv(dataframe_path / 'val_df.csv')
    train_dataset = CustomDatasetWithContours(train_df)
    validation_dataset = CustomDatasetWithContours(validation_df)

    checkpoint_path = Path(config['checkpoints_dir'])
    (checkpoint_path / 'current').mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        checkpoint_path / 'current',
        mode=config['monitor_mode'],
        filename=f'epoch_{{epoch:02d}}-{{{config["metric"]}:.3f}}',
        verbose=True
    )
    trainer = pl.Trainer(
        max_epochs=config['n_epochs'],
        accelerator=config['accelerator'],
        devices=[config['device']],
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval='epoch'),
        ],
    )
    data_module = UnetDataModule(data_dir='D:/Course_work/data/generated/df_contours/', config=config)
    data_module.setup(stage='fit')
    model = UNet()
    # trainer.fit(model, datamodule=data_module, ckpt_path='../../Checkpoints/current/epoch_epoch=04-bce_dice_both=0.000-v1.ckpt')
    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()