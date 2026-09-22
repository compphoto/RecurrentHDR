### Author: Sebastian Dille, 2026

############################################################
### Update the dataset paths below to point to your datasets
############################################################


import os
import sys

sys.path.append("..")
import argparse
import math

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import transforms

import warnings

from src.extension_dataset import CropDataset, HDRDataset
from src.model.pix2pixhd import NLayerDiscriminator
from src.model.midas_net_full import MidasNet
from src.utils import current_timestamp


def set_requires_grad(module, value):
    """
    Sets the requires_grad attribute of all parameters in a module.

    Args:
        module (nn.Module): The module whose parameters' requires_grad attribute will be set.
        value (bool): The value to set for the requires_grad attribute.
    """
    for param in module.parameters():
        param.requires_grad = value


def load_extension_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Loads the extension model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint.

    Returns:
        torch.nn.Module: The loaded extension model.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ckpt = torch.hub.load_state_dict_from_url(
            checkpoint_path, map_location=device, weights_only=True, progress=True
        )
        model = MidasNet(in_c=6, out_c=3, last_residual=True)
        full_model = Extender(model)
        print(f"Loading extension model from: {checkpoint_path}")
        full_model.load_state_dict(ckpt)
        full_model.freeze()
    return full_model.generator.to(device)


class Extender(LightningModule):
    """
    Our main class for the dual-discriminator extension model.
    It encapsulates the extender, image discriminator, and residual discriminator,
    along with their respective training steps and loss calculations.
    """

    def __init__(
        self,
        model,
        lr_g=1e-4,
        lr_d_i=1e-5,
        lr_d_r=1e-5,
        delta_g=1.0,
        delta_d_img=1.0,
        delta_d_res=1.0,
        max_ev=4,
        img_log_step=250,
        predict_only=False,
    ):
        super().__init__()
        self.generator = model
        self.lr_g = lr_g
        self.lr_d_i = lr_d_i
        self.lr_d_r = lr_d_r
        self.delta_g = delta_g
        self.delta_d_img = delta_d_img
        self.delta_d_res = delta_d_res
        self.max_ev = max_ev
        self.img_log_step = img_log_step

        if not predict_only:
            # Initialize discriminators
            norm_layer = nn.InstanceNorm2d
            self.pixel_loss = lpips.LPIPS(net="vgg", version="0.1")
            self.image_discriminator = NLayerDiscriminator(
                6, norm_layer=norm_layer
            )  # Discriminator for extended image
            self.residual_discriminator = NLayerDiscriminator(
                3, norm_layer=norm_layer
            )  # Discriminator for predicted residual
            self.save_hyperparameters()
            self.initizalized = False

    def configure_optimizers(self):
        # define discriminator for extended image
        image_discriminator_params = list(self.image_discriminator.parameters())

        # define discriminator for predicted residual
        residual_discriminator_params = list(self.residual_discriminator.parameters())

        # one optimizer each for generator, image discriminator, and residual discriminator
        g_opt = optim.AdamW(self.generator.parameters(), lr=self.lr_g)
        d_img_opt = optim.AdamW(image_discriminator_params, lr=self.lr_d_i)
        d_res_opt = optim.AdamW(residual_discriminator_params, lr=self.lr_d_r)

        # Learning rate schedulers
        g_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            g_opt, T_max=self.trainer.max_epochs
        )
        d_img_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            d_img_opt, T_max=self.trainer.max_epochs
        )
        d_res_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            d_res_opt, T_max=self.trainer.max_epochs
        )
        return [g_opt, d_img_opt, d_res_opt], [
            g_scheduler,
            d_img_scheduler,
            d_res_scheduler,
        ]

    # Forward pass through the generator
    def forward(self, x):
        return self.generator(x).clamp(0, 1)

    def expose_by_ev(self, x, ev):
        """
        Expose the image by a given EV (Exposure Value).

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            ev (torch.Tensor): Exposure value tensor of shape (B,).

        Returns:
            torch.Tensor: Exposed image tensor.
        """
        batch_size = x.shape[0]
        scale = (2**ev).view(batch_size, 1, 1, 1)
        return x * scale

    def _do_step(
        self,
        input_tensor,
        target_tensor,
        residual_tensor,
        optimizer_idx=None,
    ):
        """
        Performs a single optimization step for the generator and discriminators.

        Args:
            input_tensor (torch.Tensor): Input tensor for the generator.
            target_tensor (torch.Tensor): Target tensor for the generator.
            residual_tensor (torch.Tensor): Residual tensor for the residual discriminator.
            optimizer_idx (int, optional): Index of the optimizer to use.
                0 for generator, 1 for image discriminator, 2 for residual discriminator.

        Returns:
            tuple: A tuple containing the loss, predicted tensor, and output tensor.

        """
        if optimizer_idx == 0:  # step for generator
            set_requires_grad(self.generator, True)
            set_requires_grad(self.image_discriminator, False)
            set_requires_grad(self.residual_discriminator, False)
        elif optimizer_idx == 1:  # step for discriminator
            set_requires_grad(self.generator, False)
            set_requires_grad(self.image_discriminator, True)
            set_requires_grad(self.residual_discriminator, False)
        elif optimizer_idx == 2:  # step for residual discriminator
            set_requires_grad(self.generator, False)
            set_requires_grad(self.image_discriminator, False)
            set_requires_grad(self.residual_discriminator, True)

        # Predict residual
        output_tensor = self(input_tensor)  # Normalize timesteps to [0,1]

        # add predicted residual to degraded image to get extended image
        pred = input_tensor[:, :3, :, :] + output_tensor

        # Use degraded image as conditioning for the image discriminator
        conditions = input_tensor[:, :3, :, :]

        # Calculate loss
        loss = 0
        scale = 1.0

        # generator loss
        if optimizer_idx is None or optimizer_idx == 0:
            loss += self.delta_g * (
                1.0
                * self.pixel_loss(scale * pred - 1, scale * target_tensor - 1).mean()
            )
            loss += self.delta_g * self.adversarial_loss(
                pred, target_tensor, conditions
            )

        # image discriminator loss
        if optimizer_idx == 1:
            loss += self.delta_d_img * self.discriminator_loss(
                pred, target_tensor, conditions, gamma=0.05
            )

        # residual discriminator loss
        if optimizer_idx == 2:
            loss += self.delta_d_res * self.residual_loss(
                output_tensor, residual_tensor, gamma=0.05
            )

        return loss, pred, output_tensor

    def preprocess_batch(self, batch):
        """
        Preprocesses a batch of data for training.
        Args:
            batch (tuple): A batch of data containing images and paths.
        Returns:
            tuple: Preprocessed images, degraded images, less degraded images, residuals, input tensor, and paths.
        """
        images, paths = batch
        batch_size = images.shape[0]

        # Get all max-value in the batch
        max_vals = [torch.log2(img.max()) for img in images]

        # Sample random timesteps
        t = torch.randint(-3, 3, (batch_size,), device=self.device).float()

        # max [5,4,1,1]
        # t [-3,1,2,-1]
        # exps [t,t,t,1] -> [-3,1,2,1]
        exp_sum = t + torch.tensor(max_vals, device=self.device)
        exps = torch.where(exp_sum < 1, 1, t).float()

        # Expose images by the sampled exposure values
        images_exposed = self.expose_by_ev(images, exps)

        # Generate noise
        noise = torch.randn_like(images)

        # Targets
        less_degraded_images = images_exposed.clone()
        less_degraded_images = less_degraded_images.clamp(0, 2)

        # Inputs
        degraded_images = images_exposed.clone()
        degraded_images = degraded_images.clamp(0, 1)

        # Target residuals
        residuals = less_degraded_images - degraded_images

        # Create input for the generator
        input_tensor = torch.cat([degraded_images, noise], dim=1)
        return (
            degraded_images,
            less_degraded_images,
            residuals,
            input_tensor,
            paths,
        )

    def training_step(self, batch, batch_idx):
        """
        Training step for the GAN model.
        Args:
            batch (tuple): A batch of data containing images and paths.
            batch_idx (int): Index of the batch.
        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        if not self.initizalized:
            # Move pixel loss to the correct device
            self.pixel_loss.to(self.device)
            self.initizalized = True

        # Get batch data
        batch_size = batch[0].shape[0]
        (
            degraded_images,
            less_degraded_images,
            residuals,
            input_tensor,
            paths,
        ) = self.preprocess_batch(batch)

        self._is_training_step = True
        g_opt, d_img_opt, d_res_opt = self.optimizers()

        # Generator step
        loss_g, _, _ = self._do_step(
            input_tensor, less_degraded_images, residuals, mode="train", optimizer_idx=0
        )
        g_opt.zero_grad()

        # Our dual-discriminator setup requires manual backward to avoid errors
        self.manual_backward(loss_g)
        self.clip_gradients(g_opt, gradient_clip_val=1, gradient_clip_algorithm="norm")
        g_opt.step()

        # Image Discriminator step
        loss_d_img, _, _ = self._do_step(
            input_tensor, less_degraded_images, residuals, mode="train", optimizer_idx=1
        )
        d_img_opt.zero_grad()
        self.manual_backward(loss_d_img)
        self.clip_gradients(
            d_img_opt, gradient_clip_val=1, gradient_clip_algorithm="norm"
        )
        d_img_opt.step()

        # Residual Discriminator step
        loss_d_res, pred, output_tensor = self._do_step(
            input_tensor, less_degraded_images, residuals, mode="train", optimizer_idx=2
        )
        d_res_opt.zero_grad()
        self.manual_backward(loss_d_res)
        self.clip_gradients(
            d_res_opt, gradient_clip_val=1, gradient_clip_algorithm="norm"
        )
        d_res_opt.step()

        # combine losses
        loss = (
            self.delta_g * loss_g
            + self.delta_d_img * loss_d_img
            + self.delta_d_res * loss_d_res
        )

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train_loss_g",
            loss_g,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train_loss_d_img",
            loss_d_img,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train_loss_d_res",
            loss_d_res,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )

        # Log images for visualization
        if batch_idx % self.img_log_step == 0:
            fpath = paths[0]
            prediction = pred[0]
            target = less_degraded_images[0]
            residual_pred = output_tensor[0]
            prediction = prediction.clamp(0, 2) / 2
            target = target.clamp(0, 2) / 2
            residual_pred = residual_pred.clamp(0, 2) / 2

            self.logger.log_image(
                key="rgb_gt",
                images=[target],
                caption=[f"RGB GT, {fpath}"],
            )
            self.logger.log_image(
                key="rgb_pred",
                images=[prediction],
                caption=["RGB Predicted"],
            )
            self.logger.log_image(
                key="rgb_in", images=[degraded_images[0] / 2], caption=["RGB LDR"]
            )
            self.logger.log_image(
                key="res_pred",
                images=[residual_pred],
                caption=["Residual Predicted"],
            )
            self.logger.log_image(
                key="res_gt", images=[residuals[0] / 2], caption=["Residual GT"]
            )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the GAN model.
        Args:
            batch (tuple): A batch of data containing images and paths.
            batch_idx (int): Index of the batch.
        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        if not self.initizalized:
            # Move pixel loss to the correct device
            self.pixel_loss.to(self.device)
            self.initizalized = True

        # Get batch data
        batch_size = batch[0].shape[0]
        (
            _,
            less_degraded_images,
            residuals,
            input_tensor,
            _,
        ) = self.preprocess_batch(batch)

        # Calculate loss
        loss, _, _ = self._do_step(
            input_tensor,
            less_degraded_images,
            residuals,
            optimizer_idx=None,
        )

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss

    def zeroCenteredGradientPenalty(self, samples, critics):
        """
        Computes the zero-centered gradient penalty for the discriminator.
        Args:
            samples (torch.Tensor): Samples from the discriminator.
            critics (torch.Tensor): Critic scores for the samples.
        Returns:
            torch.Tensor: The zero-centered gradient penalty.
        """
        (gradient,) = torch.autograd.grad(
            outputs=critics.sum(), inputs=samples, create_graph=True
        )
        return gradient.square().sum([1, 2, 3])

    def adversarial_loss(self, fakeSamples, realSamples, conditions):
        """Computes the adversarial loss for the GAN.
        Args:
            fakeSamples (torch.Tensor): Generated samples from the generator.
            realSamples (torch.Tensor): Real samples from the dataset.
            conditions (torch.Tensor): Conditions for the discriminator.
        Returns:
            torch.Tensor: The adversarial loss.
        """
        realSamples = realSamples.detach()

        fakeLogits, _ = self.image_discriminator(
            torch.cat([realSamples, conditions], dim=1)
        )
        realLogits, _ = self.image_discriminator(
            torch.cat([fakeSamples, conditions], dim=1)
        )

        relativisticLogits = fakeLogits - realLogits
        return nn.functional.softplus(-relativisticLogits).mean()

    def discriminator_loss(self, fakeSamples, realSamples, conditions, gamma):
        """Computes the discriminator loss for the GAN.
        Args:
            fakeSamples (torch.Tensor): Generated samples from the generator.
            realSamples (torch.Tensor): Real samples from the dataset.
            conditions (torch.Tensor): Conditions for the discriminator.
            gamma (float): Weight for the gradient penalty.
        Returns:
            torch.Tensor: The discriminator loss.
        """
        realSamples = realSamples.detach().requires_grad_(True)
        fakeSamples = fakeSamples.detach().requires_grad_(True)

        realLogits, _ = self.image_discriminator(
            torch.cat([realSamples, conditions], dim=1)
        )
        fakeLogits, _ = self.image_discriminator(
            torch.cat([fakeSamples, conditions], dim=1)
        )

        r1Penalty = self.zeroCenteredGradientPenalty(realSamples, realLogits)
        r2Penalty = self.zeroCenteredGradientPenalty(fakeSamples, fakeLogits)

        relativisticLogits = realLogits - fakeLogits
        adversarialLoss = nn.functional.softplus(-relativisticLogits).mean(
            dim=(1, 2, 3)
        )

        discriminatorLoss = adversarialLoss + (gamma / 2) * (r1Penalty + r2Penalty)
        return discriminatorLoss.mean()

    def residual_loss(self, fakeSamples, realSamples, gamma):
        """Computes the unconditional residual discriminator loss for the GAN.
        Args:
            fakeSamples (torch.Tensor): Generated samples from the generator.
            realSamples (torch.Tensor): Real samples from the dataset.
            gamma (float): Weight for the gradient penalty.
        Returns:
            torch.Tensor: The discriminator loss.
        """
        realSamples = realSamples.detach().requires_grad_(True)
        fakeSamples = fakeSamples.detach().requires_grad_(True)

        realLogits, _ = self.residual_discriminator(realSamples)
        fakeLogits, _ = self.residual_discriminator(fakeSamples)

        r1Penalty = self.zeroCenteredGradientPenalty(realSamples, realLogits)
        r2Penalty = self.zeroCenteredGradientPenalty(fakeSamples, fakeLogits)

        relativisticLogits = realLogits - fakeLogits
        adversarialLoss = nn.functional.softplus(-relativisticLogits).mean(
            dim=(1, 2, 3)
        )

        residualLoss = adversarialLoss + (gamma / 2) * (r1Penalty + r2Penalty)
        return residualLoss.mean()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag", type=str, default="training", help="tag to identify runs"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=300, help="Number of epochs to train"
    )
    parser.add_argument(
        "--img_log_step", type=int, default=250, help="Frequency of logging images"
    )
    parser.add_argument(
        "--num_devices",
        type=int,
        default=1,
        help="Number of devices to use for training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size for training",
    )
    parser.add_argument(
        "--log_frequency", type=int, default=50, help="Frequency of logging"
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=4,
        help="Number of batches to accumulate gradients over",
    )
    parser.add_argument(
        "--delta_d_img",
        type=float,
        default=0.1,
        help="Weight for image discriminator loss",
    )
    parser.add_argument(
        "--delta_d_res",
        type=float,
        default=0.1,
        help="Weight for residual discriminator loss",
    )
    parser.add_argument(
        "--lr_g",
        type=float,
        default=1e-4,
        help="LR for generator",
    )
    parser.add_argument(
        "--lr_d_res",
        type=float,
        default=1e-5,
        help="LR for residual discriminator",
    )
    parser.add_argument(
        "--lr_d_img",
        type=float,
        default=1e-5,
        help="LR for image discriminator",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--last_run_dir",
        type=str,
        default=None,
        help="Directory of the last run to resume from",
    )
    args = parser.parse_args()

    # Initialize model
    model = MidasNet(in_c=6, out_c=3, last_residual=True)

    max_val = 1.0

    full_model = nn.Sequential(
        model,
        nn.Hardtanh(
            min_val=0.0, max_val=max_val, inplace=False
        ),  # Ensure output is in [0, 1] range
    )

    # Create the diffusion module
    gan_module = Extender(
        model,
        lr_g=args.lr_g,
        lr_d_i=args.lr_d_img,
        lr_d_r=args.lr_d_res,
        delta_g=100.0,
        delta_d_img=args.delta_d_img,
        delta_d_res=args.delta_d_res,
        max_ev=2,
        img_log_step=args.img_log_step,
    )

    # Define dataset and transforms
    hdr_transform = transforms.Compose(
        [
            # transforms.ToTensor(),
            transforms.RandomCrop((480, 480)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((384, 384)),
        ]
    )
    raw_transform = transforms.Compose(
        [
            # transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((384, 384)),
        ]
    )

    #######################################################################################
    # Update the dataset paths below to point to your datasets                            #
    #######################################################################################

    hdr_dataset = HDRDataset(
        root_dir="path/to/hdrt/dataset",
        transform=hdr_transform,
    )

    s2r_dataset = HDRDataset(
        root_dir="/path/to/s2r_hdr/dataset",
        transform=hdr_transform,
    )
    hdrreal_dataset = HDRDataset(
        root_dir="/path/to/hdr_real/dataset",
        transform=hdr_transform,
    )

    raw_dataset = CropDataset(
        root_dir="/path/to/preprocessed/raw/dataset",
        transform=raw_transform,
    )

    dataset = ConcatDataset([hdr_dataset, hdrreal_dataset, s2r_dataset, raw_dataset])

    # get splits
    train_split = int(math.floor(0.9 * len(dataset)))
    val_split = len(dataset) - train_split

    # Define data loaders
    train_set, val_set = random_split(
        dataset, [train_split, val_split], generator=torch.Generator().manual_seed(42)
    )
    dataloader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=7,
        pin_memory=True,
    )
    valloader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=7,
        pin_memory=True,
    )

    #######################################################################################
    # Update the path to your checkpoints directory below                                 #
    #######################################################################################

    # Initialize logger and checkpoint callback
    logger = WandbLogger(project="recurrent_hdr", name="extension", dir="wandb/logs")

    if args.resume:
        checkpoint_dir = args.last_run_dir
    else:
        checkpoint_dir = "/path/to/checkpoints/extension_{}_{}_{}/".format(
            args.tag, args.mode, current_timestamp()
        )

    # Save checkpoints every 5 epochs
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        dirpath=checkpoint_dir,
        every_n_epochs=5,
        save_last=True,
    )

    # Check if a last checkpoint exists to resume from
    last_checkpoint_path = os.path.join(checkpoint_dir, "last.ckpt")
    if os.path.exists(last_checkpoint_path):
        resume_from_checkpoint = last_checkpoint_path
    else:
        resume_from_checkpoint = None

    # Train the model
    trainer = Trainer(
        callbacks=checkpoint_callback,
        default_root_dir=os.getcwd(),
        devices=args.num_devices,
        strategy="ddp_find_unused_parameters_true" if args.num_devices > 1 else "auto",
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_frequency,
        logger=logger,
    )

    # Enable benchmark mode for improved performance
    trainer.benchmark = True

    # Start training
    trainer.fit(gan_module, dataloader, valloader, ckpt_path=resume_from_checkpoint)
