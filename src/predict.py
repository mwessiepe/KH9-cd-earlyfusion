import os
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_bounds

from lightning.pytorch import Trainer

from src.datamodule import KH9CdDataModule
from src.task import CustomSemanticSegmentationTask, ChangeStarFarSegTask

def predict(old_images_dir, new_images_dir, bag_buildings_dir,
            experiment_dir, batch_size, patch_size, num_dataloader_workers,
            val_split_pct, test_split_pct, checkpoint_name, rois, aoi, task, predictions_dir):
    torch.set_float32_matmul_precision('medium')

    # Init datamodule with predict dataset
    datamodule = KH9CdDataModule(
        old_images_dir=old_images_dir,
        new_images_dir=new_images_dir,
        bag_buildings_dir=bag_buildings_dir,
        batch_size=batch_size,
        num_workers=num_dataloader_workers,
        patch_size=patch_size,
        val_split_pct=val_split_pct,
        test_split_pct=test_split_pct,
        rois=rois,
        aoi=aoi,
    )

    datamodule.setup("predict")
    os.makedirs(predictions_dir, exist_ok=True)
    checkpoint_path = os.path.join(experiment_dir, checkpoint_name)
    if task == 'baseline':
        task = CustomSemanticSegmentationTask.load_from_checkpoint(checkpoint_path)
        task.predictions_dir = predictions_dir
    elif task == 'ChangeStarFarSeg':
        task = ChangeStarFarSegTask.load_from_checkpoint(checkpoint_path)
        task.predictions_dir = predictions_dir
    task.eval()

    trainer = Trainer(
        accelerator="gpu", 
        devices=[0],
        )
    trainer.predict(model=task, datamodule=datamodule)

