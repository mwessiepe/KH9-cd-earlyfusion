import os
import rasterio
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torchmetrics import JaccardIndex, F1Score

from torchgeo.models import ChangeStarFarSeg
from torchgeo.trainers import SemanticSegmentationTask
from torchgeo.datasets import unbind_samples

from .augmentations import BitemporalAugmentationModule


class CustomSemanticSegmentationTask(SemanticSegmentationTask):
    def __init__(self, *args, **kwargs):
        # if 'ignore_index' in kwargs:
        #     del kwargs['ignore']
        super().__init__(*args, **kwargs)

        self.train_augmentations = BitemporalAugmentationModule()
    
    def plot(self, sample: dict):
        image = sample["image"].squeeze(0)  # [4, H, W]
        gt_mask = sample["mask"].squeeze(0).numpy()  # [H, W]

        image1 = image[0].numpy()  # old panchromatic
        image2 = image[1:4].numpy().transpose(1, 2, 0)  # new RGB

        image2 = torch.clamp(torch.tensor(image2) / 250.0, min=0, max=1).numpy()

        pred_mask = sample["prediction"].numpy()
        
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(image1, cmap='gray')
        axs[0].axis("off")
        axs[0].set_title("Image 1")
        axs[1].imshow(image2)
        axs[1].axis("off")
        axs[1].set_title("Image 2")
        axs[2].imshow(gt_mask, cmap="gray")
        axs[2].axis("off")
        axs[2].set_title("Ground Truth")
        axs[3].imshow(pred_mask, cmap="gray")
        axs[3].axis("off")
        axs[3].set_title("Prediction")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def training_step(self, *args, **kwargs):
        batch = args[0]
        batch_idx = args[1]

        x = batch["image"]
        y = batch["mask"]

        # Move to device
        x = x.to(self.device)
        y = y.to(self.device)

        # Apply augmentations (ensure model is ready)
        if self.train_augmentations is not None:
            self.train_augmentations = self.train_augmentations.to(self.device)
            x, y = self.train_augmentations(x, y)
            batch["image"] = x
            batch["mask"] = y

        y = y.squeeze(1)  # For CrossEntropyLoss: [B, H, W]

        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_metrics(y_hat_hard, y)

        if batch_idx < 10:
            batch["prediction"] = y_hat_hard
            for key in ["image", "mask", "prediction"]:
                batch[key] = batch[key].cpu()
            sample = unbind_samples(batch)[0]
            fig = self.plot(sample)
            # plt.show()
            summary_writer = self.logger.experiment
            summary_writer.add_figure(
                f"image/train/{batch_idx}", fig, global_step=self.global_step
            )
            plt.close()

        return loss


    # The only difference between this code and the same from SemanticSegmentationTask is our redirect to use our own plotting function
    def validation_step(self, *args, **kwargs):
        batch = args[0]
        batch_idx = args[1]
        x = batch["image"]
        y = batch["mask"].squeeze(1)
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.criterion(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_metrics(y_hat_hard, y)

        if batch_idx < 10:
            batch["prediction"] = y_hat_hard
            for key in ["image", "mask", "prediction"]:
                batch[key] = batch[key].cpu()
            sample = unbind_samples(batch)[0]
            fig = self.plot(sample)
            summary_writer = self.logger.experiment
            summary_writer.add_figure(
                f"image/val/{batch_idx}", fig, global_step=self.global_step
            )
            plt.close()

    def test_step(self, *args, **kwargs):
        batch = args[0]
        batch_idx = args[1]

        x = batch["image"]
        y = batch["mask"].squeeze(1)
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        self.test_metrics(y_hat_hard, y)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)

        self.logged_test_images = getattr(self, 'logged_test_images', 0)

        # Plot total of 50 samples
        if self.logged_test_images < 50:
            batch["prediction"] = y_hat_hard
            for key in ["image", "mask", "prediction"]:
                batch[key] = batch[key].cpu()
            sample = unbind_samples(batch)[0]
            fig = self.plot(sample)

            summary_writer = self.logger.experiment
            summary_writer.add_figure(
                f"image/test/{self.logged_test_images}", fig, global_step=self.global_step
            )
            plt.close()
            self.logged_test_images += 1

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch["image"]
        y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)  # [B, H, W]

        # Move prediction to CPU
        preds = y_hat_hard.cpu()

        # Define the directory for saving predictions
        predictions_dir = "C:\masterarbeit\code\predictions"
        os.makedirs(predictions_dir, exist_ok=True)

        for i, sample in enumerate(unbind_samples({**batch, "prediction": preds})):
            prediction = sample["prediction"].numpy().astype("uint8")
            bounds = sample["bounds"]
            crs = sample["crs"]

            # Compute the affine transformation
            transform = from_bounds(bounds.minx, bounds.miny, bounds.maxx, bounds.maxy,
                                    prediction.shape[1], prediction.shape[0])

            # Build file path
            output_path = os.path.join(predictions_dir, f"pred_{batch_idx:04}_{i}.tif")

            # Save prediction as GeoTIFF
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=prediction.shape[0],
                width=prediction.shape[1],
                count=1,
                dtype="uint8",
                crs=crs,
                transform=transform,
            ) as dst:
                dst.write(prediction, 1)

        return preds


class ChangeStarFarSegTask(LightningModule):
    def __init__(self, backbone='resnet50', classes=1, lr=1e-3, backbone_pretrained=True, predictions_dir=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = ChangeStarFarSeg(
            backbone=backbone,
            classes=classes,
            backbone_pretrained=backbone_pretrained
        )
        self.train_criterion = nn.BCEWithLogitsLoss()
        # self.eval_criterion = nn.BCELoss()
        
        # Metrics for each phase
        self.train_iou = JaccardIndex(task='binary')
        self.train_f1 = F1Score(task='binary', average='macro')
        self.val_iou = JaccardIndex(task='binary')
        self.val_f1 = F1Score(task='binary', average='macro')
        self.test_iou = JaccardIndex(task='binary')
        self.test_f1 = F1Score(task='binary', average='macro')
        self.predictions_dir = predictions_dir
    
    def forward(self, x: torch.Tensor) -> dict:
        return self.model(x)
    
    def plot(self, sample: dict):
        images = sample["image"]
        image1 = images[0, 0, :, :].numpy()
        image2 = images[1].numpy().transpose(1, 2, 0)
        image2 = torch.clamp(torch.tensor(image2) / 250, min=0, max=1).numpy()

        gt_mask = sample["mask"].numpy()
        pred_mask = sample["prediction"].numpy()
        
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(image1, cmap='gray')
        axs[0].axis("off")
        axs[0].set_title("Image 1")
        axs[1].imshow(image2)
        axs[1].axis("off")
        axs[1].set_title("Image 2")
        axs[2].imshow(gt_mask, cmap="gray")
        axs[2].axis("off")
        axs[2].set_title("Ground Truth")
        axs[3].imshow(pred_mask, cmap="gray")
        axs[3].axis("off")
        axs[3].set_title("Prediction")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def training_step(self, *args, **kwargs):
        batch = args[0]
        batch_idx = args[1]
        
        x = batch["image"]
        y = batch["mask"]
        y_hat_dict = self.forward(x)
        # Average the two output logits from ChangeMixin
        change_logits = y_hat_dict["bi_change_logit"].mean(dim=1)
        
        loss = self.train_criterion(change_logits, y.unsqueeze(1).float())
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        
        y_hat_hard = (torch.sigmoid(change_logits) > 0.5).long().squeeze(1)
        self.train_iou.update(y_hat_hard, y.long())
        self.train_f1.update(y_hat_hard, y.long())
        
        if batch_idx < 10:
            batch["prediction"] = y_hat_hard
            for key in ["image", "mask", "prediction"]:
                batch[key] = batch[key].cpu()
            sample = unbind_samples(batch)[0]
            fig = self.plot(sample)
            summary_writer = self.logger.experiment
            summary_writer.add_figure(f"image/train/{batch_idx}", fig, global_step=self.global_step)
            plt.close(fig)
        
        return loss
    
    def validation_step(self, *args, **kwargs):
        batch = args[0]
        batch_idx = args[1]
        
        x = batch["image"]
        y = batch["mask"][:, 0, 0, ...]
        
        y_hat_dict = self.forward(x)
        change_prob = y_hat_dict["change_prob"]
        # loss = self.eval_criterion(change_prob, y.unsqueeze(1).float())
        # self.log("val_loss", loss, on_step=False, on_epoch=True)
        
        y_hat_hard = (change_prob > 0.5).long().squeeze(1)
        self.val_iou.update(y_hat_hard, y.long())
        self.val_f1.update(y_hat_hard, y.long())
        
        if batch_idx < 10:
            batch["prediction"] = y_hat_hard
            for key in ["image", "mask", "prediction"]:
                batch[key] = batch[key].cpu()
            sample = unbind_samples(batch)[0]
            fig = self.plot(sample)
            summary_writer = self.logger.experiment
            summary_writer.add_figure(f"image/val/{batch_idx}", fig, global_step=self.global_step)
            plt.close(fig)

    def test_step(self, *args, **kwargs):
        batch = args[0]
        batch_idx = args[1]
        
        x = batch["image"]
        y = batch["mask"][:, 0, 0, ...]
        
        y_hat_dict = self.forward(x)
        change_prob = y_hat_dict["change_prob"]
        # loss = self.eval_criterion(change_prob, y.unsqueeze(1).float())
        # self.log("test_loss", loss, on_step=False, on_epoch=True)
        
        y_hat_hard = (change_prob > 0.5).long().squeeze(1)
        self.test_iou.update(y_hat_hard, y.long())
        self.test_f1.update(y_hat_hard, y.long())
        self.log("test_iou", self.test_iou.compute())
        self.log('test_f1', self.test_f1.compute(), on_step=False, on_epoch=True)
        # self.log_dict({
        #     "test_iou": self.test_iou.compute(),
        #     "test_f1": self.test_f1.compute()
        # }, prog_bar=True)
        
        self.logged_test_images = getattr(self, 'logged_test_images', 0)
        if self.logged_test_images < 50:
            batch["prediction"] = y_hat_hard
            for key in ["image", "mask", "prediction"]:
                batch[key] = batch[key].cpu()
            sample = unbind_samples(batch)[0]
            fig = self.plot(sample)
            summary_writer = self.logger.experiment
            summary_writer.add_figure(f"image/test/{self.logged_test_images}", fig, global_step=self.global_step)
            plt.close(fig)
            self.logged_test_images += 1

    def predict_step(self, batch, batch_idx, dataloader_idx=0 ):
        x = batch["image"]
        y_hat_dict = self(x)
        change_prob = y_hat_dict["change_prob"]
        y_hat_hard = (change_prob > 0.5).long().squeeze(1)
        preds = y_hat_hard.cpu()
        
        os.makedirs(self.predictions_dir, exist_ok=True)
        for i, sample in enumerate(unbind_samples({**batch, "prediction": preds})):
            prediction = sample["prediction"].numpy().astype("uint8")
            bounds = sample["bounds"]
            crs = sample["crs"]
            transform = from_bounds(bounds.minx, bounds.miny, bounds.maxx, bounds.maxy,
                                      prediction.shape[1], prediction.shape[0])
            output_path = os.path.join(self.predictions_dir, f"pred_{batch_idx:04}_{i}.tif")
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=prediction.shape[0],
                width=prediction.shape[1],
                count=1,
                dtype="uint8",
                crs=crs,
                transform=transform,
            ) as dst:
                dst.write(prediction, 1)
        return preds

    def configure_optimizers(self):
        print(self.hparams)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer