import os
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_bounds
from rasterio.merge import merge

from lightning.pytorch import Trainer

from src.datamodule import KH9CdDataModule
from src.task import CustomSemanticSegmentationTask, ChangeStarFarSegTask

import glob
from rasterio.coords import BoundingBox
from rasterio.windows import from_bounds as window_from_bounds


def merge_predictions(predictions_dir, output_path):
    import os
    import glob
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.windows import from_bounds as window_from_bounds
    from tqdm import tqdm

    pred_files = sorted(glob.glob(os.path.join(predictions_dir, "pred_*.tif")))
    if not pred_files:
        raise RuntimeError(f"No prediction tiles found in {predictions_dir}")

    # Determine global bounds and metadata
    bounds_list = []
    crs = None
    for f in pred_files:
        with rasterio.open(f) as src:
            bounds_list.append(src.bounds)
            if crs is None:
                crs = src.crs

    minx = min(b.left for b in bounds_list)
    miny = min(b.bottom for b in bounds_list)
    maxx = max(b.right for b in bounds_list)
    maxy = max(b.top for b in bounds_list)

    with rasterio.open(pred_files[0]) as src:
        res_x, res_y = src.res
        patch_height, patch_width = src.height, src.width

    out_width = int(np.ceil((maxx - minx) / res_x))
    out_height = int(np.ceil((maxy - miny) / res_y))
    out_transform = from_bounds(minx, miny, maxx, maxy, out_width, out_height)

    # Initialize accumulation arrays
    logit_sum = np.zeros((3, out_height, out_width), dtype=np.float32)
    weight_sum = np.zeros((out_height, out_width), dtype=np.float32)

    for f in tqdm(pred_files, desc="Merging patches"):
        with rasterio.open(f) as src:
            class0 = src.read(1)  # logit for class 0
            class1 = src.read(2)  # logit for class 1
            class2 = src.read(3)  # logit for class 2
            weight = src.read(4)

            bounds = src.bounds
            window = window_from_bounds(
                bounds.left, bounds.bottom,
                bounds.right, bounds.top,
                transform=out_transform
            ).round_offsets().round_lengths()

            row_off = int(window.row_off)
            col_off = int(window.col_off)

            h, w = class0.shape
            logit_sum[0, row_off:row_off+h, col_off:col_off+w] += class0
            logit_sum[1, row_off:row_off+h, col_off:col_off+w] += class1
            logit_sum[2, row_off:row_off+h, col_off:col_off+w] += class2
            weight_sum[row_off:row_off+h, col_off:col_off+w] += weight

    # Normalize logits
    weight_sum_safe = weight_sum + 1e-6  # prevent divide-by-zero
    logit_sum[0] /= weight_sum_safe
    logit_sum[1] /= weight_sum_safe
    logit_sum[2] /= weight_sum_safe

    # Compute final prediction via softmax → argmax
    exp_logits = np.exp(logit_sum - np.max(logit_sum, axis=0, keepdims=True))  # [3, H, W]
    probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
    final_pred = np.argmax(probs, axis=0).astype(np.uint8)  # values: 0, 1, 2

    # Save final prediction
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=final_pred.shape[0],
        width=final_pred.shape[1],
        count=1,
        dtype="uint8",
        crs=crs,
        transform=out_transform,
        compress="ZSTD",
    ) as dst:
        dst.write(final_pred, 1)

    print(f"Merged prediction saved to: {output_path}")




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
        overlap=64  # Add overlap parameter
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

    trainer = Trainer()
    trainer.predict(model=task, datamodule=datamodule)
    
    # Merge predictions with blending
    output_path = os.path.join(predictions_dir, "merged_prediction.tif")
    merge_predictions(predictions_dir, output_path)

