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
    def __init__(self, *args, predictions_dir=None, **kwargs):
        # Remove unexpected keys that may come from checkpoint/config
        kwargs.pop("ignore", None)
        super().__init__(*args, **kwargs)
        self.predictions_dir = predictions_dir
        self.train_augmentations = BitemporalAugmentationModule()
        self.weight_mask = None  # Will be initialized during prediction
    
    def create_weight_mask(self, patch_size):
        """Create a weight mask for blending overlapping regions."""
        overlap = 64  # Default overlap size
        weight = torch.ones((patch_size, patch_size), dtype=torch.float32)
        
        # Create gradual blending in overlap regions
        for i in range(overlap):
            # Linear weight from 0 to 1 for left and top edges
            alpha = i / overlap
            weight[i, :] *= alpha
            weight[:, i] *= alpha
            # Linear weight from 1 to 0 for right and bottom edges
            weight[-(i+1), :] *= alpha
            weight[:, -(i+1)] *= alpha
        
        return weight

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
        x = batch["image"]  # shape: [B, C, H, W]

        # Initialize weight mask if not already done
        if self.weight_mask is None:
            patch_size = x.shape[-1]  # Assuming square patches
            self.weight_mask = self.create_weight_mask(patch_size).to(x.device)  # shape: [H, W]

        y_hat = self(x)  # shape: [B, 3, H, W] — raw logits
        weighted_logits = y_hat * self.weight_mask.unsqueeze(0).unsqueeze(0)  # [B, 3, H, W]

        for i, sample in enumerate(unbind_samples({**batch})):
            logits = weighted_logits[i]  # shape: [3, H, W]
            weight = self.weight_mask.cpu().numpy().astype("float32")
            bounds = sample["bounds"]
            crs = sample["crs"]

            transform = from_bounds(bounds.minx, bounds.miny, bounds.maxx, bounds.maxy,
                                    weight.shape[1], weight.shape[0])

            output_path = os.path.join(self.predictions_dir, f"pred_{batch_idx:04}_{i}.tif")
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=weight.shape[0],
                width=weight.shape[1],
                count=4,  # 3 classes + 1 weight mask
                dtype="float32",
                crs=crs,
                transform=transform,
            ) as dst:
                for c in range(3):
                    dst.write(logits[c].cpu().numpy().astype("float32"), c + 1)
                dst.write(weight, 4)

        return weighted_logits.cpu()



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
        # x: [B, 4, H, W] -> [B, 2, C, H, W]
        # Split into t1 (1ch) and t2 (3ch)
        x_t1 = x[:, 0:1, :, :]            # [B, 1, H, W]
        x_t2 = x[:, 1:4, :, :]            # [B, 3, H, W]

        # Duplicate t1 to 3 channels
        x_t1_3ch = x_t1.repeat(1, 3, 1, 1)  # [B, 3, H, W]

        # Stack bitemporal into shape [B, 2, 3, H, W]
        x_bitemp = torch.stack([x_t1_3ch, x_t2], dim=1)
        return self.model(x_bitemp)

    
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from lightning.pytorch import LightningModule

class DualEncoderChangeDetectionTask(LightningModule):
    def __init__(
        self,
        encoder_old_name="resnet18",
        encoder_new_name="resnet18",
        num_classes=2,
        lr=1e-3,
        weights=True,
        ignore_index=99,
    ):
        super().__init__()
        self.save_hyperparameters()
        print(f"[DualEncoder] Init: encoder_old_name={encoder_old_name}, encoder_new_name={encoder_new_name}, num_classes={num_classes}, lr={lr}, weights={weights}, ignore_index={ignore_index}")
        self.encoder_old = models.__dict__[encoder_old_name](
            pretrained=weights
        )
        self.encoder_old.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_new = models.__dict__[encoder_new_name](
            pretrained=weights
        )
        self.encoder_new.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Feature fusion and decoder
        self.fuse_conv = nn.Conv2d(512 + 512, 256, kernel_size=3, padding=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, 1)
        )
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        from .augmentations import BitemporalAugmentationModule
        self.train_augmentations = BitemporalAugmentationModule()

    def forward(self, x):
        print(f"[DualEncoder] Forward: x.shape={x.shape}")
        old_img = x[:, 0:1, :, :]
        new_img = x[:, 1:4, :, :]
        print(f"[DualEncoder] old_img.shape={old_img.shape}, new_img.shape={new_img.shape}")
        if hasattr(self.encoder_old, 'forward_features'):
            old_feat = self.encoder_old.forward_features(old_img)
        else:
            old = self.encoder_old.conv1(old_img)
            old = self.encoder_old.bn1(old)
            old = self.encoder_old.relu(old)
            old = self.encoder_old.layer1(old)
            old = self.encoder_old.layer2(old)
            old = self.encoder_old.layer3(old)
            old = self.encoder_old.layer4(old)
            old_feat = self.encoder_old.avgpool(old)
        # Encode new image
        if hasattr(self.encoder_new, 'forward_features'):
            new_feat = self.encoder_new.forward_features(new_img)
        else:
            new = self.encoder_new.conv1(new_img)
            new = self.encoder_new.bn1(new)
            new = self.encoder_new.relu(new)
            new = self.encoder_new.layer1(new)
            new = self.encoder_new.layer2(new)
            new = self.encoder_new.layer3(new)
            new = self.encoder_new.layer4(new)
            new_feat = self.encoder_new.avgpool(new)
        print(f"[DualEncoder] old_feat.shape={old_feat.shape}, new_feat.shape={new_feat.shape}")
        if old_feat.ndim == 4 and old_feat.shape[2] == 1 and old_feat.shape[3] == 1:
            old_feat = old_feat.squeeze(-1).squeeze(-1)
        if new_feat.ndim == 4 and new_feat.shape[2] == 1 and new_feat.shape[3] == 1:
            new_feat = new_feat.squeeze(-1).squeeze(-1)
        if old_feat.ndim == 2:
            old_feat = old_feat.unsqueeze(-1).unsqueeze(-1)
        if new_feat.ndim == 2:
            new_feat = new_feat.unsqueeze(-1).unsqueeze(-1)
        print(f"[DualEncoder] old_feat (after squeeze/unsqueeze): {old_feat.shape}, new_feat: {new_feat.shape}")
        fused = torch.cat([old_feat, new_feat], dim=1)
        print(f"[DualEncoder] fused.shape={fused.shape}")
        fused = self.fuse_conv(fused)
        print(f"[DualEncoder] fused after fuse_conv: {fused.shape}")
        fused_up = F.interpolate(fused, size=x.shape[2:], mode="bilinear", align_corners=False)
        print(f"[DualEncoder] fused_up.shape={fused_up.shape}")
        out = self.decoder(fused_up)
        print(f"[DualEncoder] decoder out.shape={out.shape}, min={out.min().item()}, max={out.max().item()}")
        return out

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

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["mask"]
        print(f"[DualEncoder] training_step: x.shape={x.shape}, y.shape={y.shape}, batch_idx={batch_idx}")
        x = x.to(self.device)
        y = y.to(self.device)
        if self.train_augmentations is not None:
            self.train_augmentations = self.train_augmentations.to(self.device)
            x, y = self.train_augmentations(x, y)
            batch["image"] = x
            batch["mask"] = y
        y = y.squeeze(1)
        logits = self(x)
        print(f"[DualEncoder] logits.shape={logits.shape}, logits.min={logits.min().item()}, logits.max={logits.max().item()}")
        print(f"[DualEncoder] y unique: {torch.unique(y)}")
        loss = self.criterion(logits, y)
        print(f"[DualEncoder] loss={loss.item()}")
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        # Optionally log images
        if batch_idx < 10 and hasattr(self, "logger") and self.logger is not None:
            from torchgeo.datasets import unbind_samples
            batch["prediction"] = logits.argmax(dim=1)
            for key in ["image", "mask", "prediction"]:
                batch[key] = batch[key].cpu()
            sample = unbind_samples(batch)[0]
            fig = self.plot(sample)
            summary_writer = self.logger.experiment
            summary_writer.add_figure(
                f"image/train/{batch_idx}", fig, global_step=self.global_step
            )
            plt.close()
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["mask"].squeeze(1)
        logits = self(x)
        print(f"[DualEncoder] [VAL] logits.shape={logits.shape}, min={logits.min().item()}, max={logits.max().item()}")
        print(f"[DualEncoder] [VAL] y unique: {torch.unique(y)}")
        loss = self.criterion(logits, y)
        print(f"[DualEncoder] [VAL] loss={loss.item()}")
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        # Optionally log images
        if batch_idx < 10 and hasattr(self, "logger") and self.logger is not None:
            from torchgeo.datasets import unbind_samples
            batch["prediction"] = logits.argmax(dim=1)
            for key in ["image", "mask", "prediction"]:
                batch[key] = batch[key].cpu()
            sample = unbind_samples(batch)[0]
            fig = self.plot(sample)
            summary_writer = self.logger.experiment
            summary_writer.add_figure(
                f"image/val/{batch_idx}", fig, global_step=self.global_step
            )
            plt.close()

    def test_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["mask"].squeeze(1)
        logits = self(x)
        print(f"[DualEncoder] [TEST] logits.shape={logits.shape}, min={logits.min().item()}, max={logits.max().item()}")
        print(f"[DualEncoder] [TEST] y unique: {torch.unique(y)}")
        loss = self.criterion(logits, y)
        print(f"[DualEncoder] [TEST] loss={loss.item()}")
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        # Optionally log images
        self.logged_test_images = getattr(self, 'logged_test_images', 0)
        if self.logged_test_images < 50 and hasattr(self, "logger") and self.logger is not None:
            from torchgeo.datasets import unbind_samples
            batch["prediction"] = logits.argmax(dim=1)
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

    def predict_step(self, batch, batch_idx, dataloader_idx=0 ):
        x = batch["image"]
        logits = self(x)
        print(f"[DualEncoder] [PREDICT] logits.shape={logits.shape}, min={logits.min().item()}, max={logits.max().item()}")
        y_hat_hard = logits.argmax(dim=1)
        print(f"[DualEncoder] [PREDICT] y_hat_hard unique: {torch.unique(y_hat_hard)}")
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
        print(f"[DualEncoder] configure_optimizers: lr={self.lr}")
        return torch.optim.Adam(self.parameters(), lr=self.lr)
