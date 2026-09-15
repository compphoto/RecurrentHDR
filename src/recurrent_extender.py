### Author: Sebastian Dille, 2026

############################################################
### Update the dataset paths below to point to your datasets
############################################################

import os
import sys

sys.path.append("../src")
import argparse
import math


import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import transforms

from src.extender import Extender
from src.extension_dataset import HDRDataset
from src.model.midas_net_full import MidasNet
from src.utils import current_timestamp


class RecExtender(Extender):
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
        stages=3,
    ):
        super().__init__(
            model,
            lr_g,
            lr_d_i,
            lr_d_r,
            delta_g,
            delta_d_img,
            delta_d_res,
            max_ev,
            img_log_step,
            predict_only,
        )
        self.stages = stages

    def preprocess_batch(self, batch):
        """
        Preprocesses a batch of data for training.
        Args:
            batch (tuple): A batch of data containing images and paths.
        Returns:
            tuple: Preprocessed images, degraded images, less degraded images, residuals, input tensor, and paths.
        """
        images, paths = batch

        # Get max-value for the batch
        max_val = 2**self.stages

        # avoid div by 0, 1e-5 is below 1/65536 = minimum possible value for 16bit
        images_exposed = images.clamp(1e-5, None).clone()

        for d, i in enumerate(images_exposed):
            if i.max().item() > max_val:
                clip_val = np.random.uniform(max_val, i.max().item())
                i = torch.clamp(i, 0, clip_val)
            # set image max to max_val
            images_exposed[d] = i * max_val / i.max().item()

        # Generate noise
        noise = torch.randn_like(images)

        less_degraded_images = []
        degraded_images = []
        residuals = []

        # prepare input and targets for all stages
        for _ in range(self.stages):
            # targets
            less_degraded_images.append(images_exposed.clone())

            # inputs
            stage_max = images_exposed.amax(dim=(1, 2, 3), keepdim=True) / 2
            images_exposed = torch.clamp(
                images_exposed, torch.zeros_like(stage_max), stage_max
            )
            degraded_images.append(images_exposed.clone())

        # reverse the order of the images so that the most degraded image is first
        # renormalize each to [0, 1] for the generator
        degraded_images = list(reversed([img / img.max() for img in degraded_images]))
        less_degraded_images = list(
            reversed([img * 2 / img.max() for img in less_degraded_images])
        )

        # Target residuals
        residuals = [
            target - input
            for target, input in zip(less_degraded_images, degraded_images)
        ]

        # Create input for the generator
        input_tensor = torch.cat([degraded_images[0], noise], dim=1)
        return (
            degraded_images,
            less_degraded_images,
            residuals,
            input_tensor,
            paths,
        )

    def multistep_inference(self, opt, replay_buffer, targets, residuals, opt_stage=0):
        """
        Performs multistep inference for the GAN model.
        Args:
            opt (torch.optim.Optimizer): The optimizer for the current stage.
            replay_buffer (list): A list of input tensors for each stage.
            targets (list): A list of target tensors for each stage.
            residuals (list): A list of residual tensors for each stage.
            opt_stage (int): The index of the current optimizer.
                (0 for generator, 1 for image discriminator, 2 for residual discriminator).
        Returns:
            float: The total loss for the multistep inference.
        """
        opt.zero_grad()
        grad_buffer = torch.zeros_like(replay_buffer[0]).to(self.device)
        loss = 0

        # Iterate through each stage
        for i in reversed(range(self.stages)):
            # Forward pass through the generator
            x_i_prev = replay_buffer[i].to(self.device).detach().requires_grad_(True)

            target_i = targets[i].to(self.device)
            residual_i = residuals[i].to(self.device)

            # Compute loss and future gradient
            loss_i, _, _ = self._do_step(
                x_i_prev, target_i, residual_i, mode="train", optimizer_idx=opt_stage
            )
            self.manual_backward(loss_i)

            if opt_stage < 2:
                # Accumulate gradients from loss_i and future steps
                x_i_prev_grad = (
                    x_i_prev.grad + grad_buffer
                    if x_i_prev.grad is not None
                    else grad_buffer
                )

                # Propagate gradients to G's parameters
                x_i_prev.backward(x_i_prev_grad)
                grad_buffer = x_i_prev.grad.detach()  # Store for previous step

            # Accumulate total loss and print memory stats
            loss += loss_i.item()

        # Update weights using accumulated gradients
        opt.step()
        return loss

    def training_step(self, batch, batch_idx):
        if not self.initizalized:
            self.pixel_loss.to(self.device)
            self.initizalized = True
        """
        Training step for the GAN model.
        Args:
            batch (tuple): A batch of data containing images and paths.
            batch_idx (int): Index of the batch.
        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        # Get batch data
        batch_size = batch[0].shape[0]
        (
            _,
            less_degraded_images,
            residuals,
            input_tensor,
            paths,
        ) = self.preprocess_batch(batch)

        self._is_training_step = True
        g_opt, d_img_opt, d_res_opt = self.optimizers()

        # Initialize the current input with the LDR image
        replay_buffer = [input_tensor]  # Initial input
        outputs = []
        predictions = []

        gt_pass_prob = np.random.rand()
        if gt_pass_prob < 0.5:
            # use ground truth from previous step as input to increase stability
            for j in range(self.stages):
                # get last input
                input_t = replay_buffer[-1]
                # use gt instead of prediction
                pred = less_degraded_images[j][:, :3, :, :].clone()
                # create new input
                new_input_t = torch.cat(
                    [(pred / 2).clip(0, 1), input_t[:, 3:, :, :].clone()], dim=1
                )
                replay_buffer.append(new_input_t)
                outputs.append(residuals[j].clone())
                predictions.append(less_degraded_images[j][:, :3, :, :].clone())

        else:
            with torch.no_grad():
                for i in range(self.stages):
                    input_t = replay_buffer[-1]
                    output_tensor = self(input_t)
                    pred = input_t[:, :3, :, :].clone() + output_tensor
                    new_input_t = torch.cat(
                        [(pred / 2).clip(0, 1), input_t[:, 3:, :, :].clone()], dim=1
                    )
                    replay_buffer.append(new_input_t)  # Store for future stages
                    outputs.append(output_tensor.clone())
                    predictions.append(pred.clone())

        # free up GPU memory by moving the replay buffer to CPU
        replay_buffer = [x.cpu() for x in replay_buffer]
        outputs = [x.cpu() for x in outputs]
        predictions = [x.cpu() for x in predictions]

        torch.cuda.empty_cache()  # Free up memory before training

        # Generator step
        loss_g = self.multistep_inference(
            g_opt,
            replay_buffer,
            less_degraded_images,
            residuals,
            opt_stage=0,
            network_id="G",
        )

        # Image Discriminator step
        loss_d_img = self.multistep_inference(
            d_img_opt,
            replay_buffer,
            less_degraded_images,
            residuals,
            opt_stage=1,
            network_id="D_img",
        )

        # Residual Discriminator step
        loss_d_res = self.multistep_inference(
            d_res_opt,
            replay_buffer,
            less_degraded_images,
            residuals,
            opt_stage=2,
            network_id="D_res",
        )

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
            gt = torch.cat([t[0:1] for t in less_degraded_images], dim=3)
            pred_viz = torch.cat([t[0:1] for t in predictions], dim=3)
            ldr = torch.cat([t[0:1, 0:3] for t in replay_buffer[:-1]], dim=3)
            res_gt = torch.cat([t[0:1] for t in residuals], dim=3)
            res_pred = torch.cat([t[0:1] for t in outputs], dim=3)

            self.logger.log_image(
                key="rgb_gt",
                images=[gt.clamp(0, 2) / 2],
                caption=[f"RGB GT, {fpath}"],
            )
            self.logger.log_image(
                key="rgb_pred",
                images=[pred_viz.clamp(0, 2) / 2],
                caption=[f"RGB Predicted, max: {pred_viz.max():.2f}"],
            )
            self.logger.log_image(key="rgb_in", images=[ldr / 2], caption=["RGB LDR"])
            self.logger.log_image(
                key="res_pred",
                images=[res_pred.clamp(0, 1)],
                caption=[f"Residual Predicted, max: {res_pred.max():.2f}"],
            )
            self.logger.log_image(
                key="res_gt", images=[res_gt.clamp(0, 1)], caption=["Residual GT"]
            )

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the extension model.
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
            _,
        ) = self.preprocess_batch(batch)

        noise = torch.randn_like(degraded_images[-1])
        input_tensor = torch.cat([degraded_images[-1], noise], dim=1)

        # Calculate loss
        loss, _, _ = self._do_step(
            input_tensor,
            less_degraded_images[-1],
            residuals[-1],
            mode="train",
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


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
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
        "--stages",
        type=int,
        default=4,
        help="Number of exposure expansions to train",
    )
    parser.add_argument(
        "--additive_noise",
        action="store_true",
        help="apply noise to clipped area as well",
    )
    parser.add_argument(
        "--down",
        action="store_true",
        help="extend dynamic range in downward direction, using inverted images",
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
    full_model = nn.Sequential(
        model,
        nn.Hardtanh(
            min_val=0.0, max_val=1.0, inplace=False
        ),  # Ensure output is in [0, 1] range
    )

    # Create the diffusion module
    gan_module = RecExtender(
        model,
        lr_g=1e-4,
        lr_d_i=1e-5,
        lr_d_r=1e-5,
        delta_g=1.0,
        delta_d_img=0.1,
        delta_d_res=0.1,
        max_ev=4,
        img_log_step=args.img_log_step,
        additive_noise=args.additive_noise,
        down=args.down,
        stages=args.stages,
    )

    # Define dataset and transforms
    hdr_transform = transforms.Compose(
        [
            transforms.RandomCrop((480, 480)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((384, 384)),
        ]
    )
    raw_transform = transforms.Compose(
        [
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
        root_dir="path/to/hdr_real/dataset",
        transform=hdr_transform,
    )

    dataset = ConcatDataset(
        [hdr_dataset, hdrreal_dataset, s2r_dataset, s2r_dataset, s2r_dataset]
    )

    # Get splits
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
    logger = WandbLogger(
        project="recurrent_hdr", name="recurrent_extension", dir="wandb"
    )

    if args.resume:
        checkpoint_dir = args.last_run_dir
    else:
        checkpoint_dir = "checkpoints/recurrent_extension_{}/".format(
            current_timestamp()
        )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        dirpath=checkpoint_dir,
        every_n_epochs=5,
        save_last=True,
    )

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

    trainer.benchmark = True
    trainer.fit(gan_module, dataloader, valloader, ckpt_path=resume_from_checkpoint)
