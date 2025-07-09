import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from src.task import CustomSemanticSegmentationTask, ChangeStarFarSegTask
from src.datamodule import KH9CdDataModule


def test(old_images_dir, new_images_dir, bag_buildings_dir, experiment_name,
         experiment_dir, log_dir, batch_size, patch_size, num_dataloader_workers,
         val_split_pct, test_split_pct, checkpoint_name, rois, aoi, task):

    torch.set_float32_matmul_precision('medium')

    # Initialize the data module
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

    # Load the trained model from the checkpoint
    checkpoint_path = os.path.join(experiment_dir, checkpoint_name)
    if task == 'baseline':
        task = CustomSemanticSegmentationTask.load_from_checkpoint(checkpoint_path)
    elif task == 'ChangeStarFarSeg':
        task = ChangeStarFarSegTask.load_from_checkpoint(checkpoint_path)
    task.eval()

    # Initialize TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name=f"{experiment_name}_test"
    )

    # Run test with logger
    trainer = Trainer(
#        accelerator='gpu',
#        devices=[0],
        logger=tb_logger,
    )
    trainer.test(model=task, datamodule=datamodule)

    # # Compute additional metrics:
    # # Move model to eval mode and device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = task.model.to(device).eval()

    # y_preds = []
    # y_trues = []

    # for batch in tqdm(datamodule.test_dataloader()):
    #     images = batch["image"].to(device)
    #     with torch.no_grad():
    #         logits = model(images)
    #         preds = logits.argmax(dim=1).cpu().numpy()
    #     y_pred = preds.ravel()
    #     y_true = batch["mask"].cpu().numpy().ravel()

    #     mask = y_true != 99
    #     y_preds.append(y_pred[mask])
    #     y_trues.append(y_true[mask])

    # # Concatenate all predictions
    # y_preds = np.concatenate(y_preds)
    # y_trues = np.concatenate(y_trues)

    # f1 = f1_score(y_trues, y_preds, average='macro')
    # precision = precision_score(y_trues, y_preds, average='macro')
    # recall = recall_score(y_trues, y_preds, average='macro')

    # print(f"F1 Score: {f1:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
