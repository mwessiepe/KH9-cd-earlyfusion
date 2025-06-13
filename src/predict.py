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

    checkpoint_path = os.path.join(experiment_dir, checkpoint_name)
    if task == 'baseline':
        task = CustomSemanticSegmentationTask.load_from_checkpoint(checkpoint_path)
    elif task == 'ChangeStarFarSeg':
        task = ChangeStarFarSegTask.load_from_checkpoint(checkpoint_path)
        task.predictions_dir = predictions_dir
    task.eval()

    trainer = Trainer(
        accelerator="gpu", 
        devices=[0],
        )
    trainer.predict(model=task, datamodule=datamodule)


    # Output



    # output_dir = os.path.join(experiment_dir, "predictions")
    # os.makedirs(output_dir, exist_ok=True)

    # # Loop through each prediction
    # for i, batch_preds in enumerate(predictions):
    #     # TorchGeo outputs a list of batches; each batch is a dict with prediction and metadata
    #     for j, sample in enumerate(batch_preds):
    #         pred = sample['prediction'].squeeze().cpu().numpy()  # shape: [H, W]
    #         bounds = sample['bbox']  # This is a BoundingBox(minx, maxx, miny, maxy, mint, maxt)

    #         height, width = pred.shape
    #         transform = from_bounds(bounds.minx, bounds.maxx, bounds.miny, bounds.maxy, width, height)

    #         output_path = os.path.join(output_dir, f"prediction_{i}_{j}.tif")
    #         with rasterio.open(
    #             output_path,
    #             "w",
    #             driver="GTiff",
    #             height=height,
    #             width=width,
    #             count=1,
    #             dtype=pred.dtype,
    #             crs="EPSG:28992",  # Or replace with your actual CRS
    #             transform=transform,
    #         ) as dst:
    #             dst.write(pred, 1)

    #         print(f"Saved: {output_path}")

