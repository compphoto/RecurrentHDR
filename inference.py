### Inference script for the Recurrent Dynamic Range Extension
### Author: Sebastian Dille, 2026

import argparse
import glob
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ours
from src.utils import load_ldr, to2np, round_32, tile_imgs
from src.extender import load_extension_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "https://github.com/sebastian-dille/RecurrentHDR/releases/download/v1.0/model_weights.pth"


def recurrent_inference(
    generator: torch.nn.Module,
    ldr: np.ndarray,
    steps: int = 24,  # maximum number of exposure enhancement stages
    inference_size: int = 4096,  # maximum size to use for inference
) -> np.ndarray:
    """
    Performs recurrent inference on an LDR image to extend its dynamic range to HDR.

    Args:
        generator (torch.nn.Module): The neural network generator.
        ldr (np.ndarray): The input LDR image.
        steps (int, optional): The maximum number of exposure enhancement stages. Defaults to 24.
        inference_size (int, optional): The maximum size to use for inference. Defaults to 4096.

    Returns:
        np.ndarray: The resulting HDR image.
        int: The number of execution steps.
    """
    input_shape = ldr.shape[:2]
    ldr_t = torch.from_numpy(ldr).permute(2, 0, 1).unsqueeze(0).to(device)
    hdr_t = ldr_t.clone()  # create placeholder for hdr result

    # resize if image exceeds GPU capacity
    if max(ldr.shape) > inference_size:
        scale = inference_size / max(ldr.shape)
        ldr = cv2.resize(
            ldr,
            (int(ldr.shape[1] * scale), int(ldr.shape[0] * scale)),
            interpolation=cv2.INTER_LANCZOS4,
        ).clip(ldr.min(), ldr.max())

    # get processing size
    # our network requires multiples of 32 for input size
    prc_sz = (
        round_32(ldr.shape[0]),
        round_32(ldr.shape[1]),
    )

    # rescale image
    ldr = torch.from_numpy(ldr).permute(2, 0, 1).unsqueeze(0).to(device)
    ldr = torch.nn.functional.interpolate(
        ldr, size=prc_sz, mode="bicubic", align_corners=True, antialias=True
    ).clamp(ldr.min(), ldr.max())

    rgb_ldr = ldr.clone()

    # normalize to 99th percentile.
    # we remove top 1% of brightness as potential noise
    brightness_scale = np.percentile(rgb_ldr.cpu().numpy(), 99)
    rgb_ldr /= brightness_scale
    rgb_ldr = rgb_ldr.clamp(0, 1)

    executions = 0
    intermediates = []
    max_stages = steps

    # store initial LDR for comparisons
    intermediates.append(rgb_ldr.detach().cpu().permute(0, 2, 3, 1)[0].numpy())

    # iterative exposure enhancement
    # proceed while there are clipped areas, or until the maximum number of stages is reached
    while rgb_ldr.max() >= 0.9 and executions < max_stages:
        # initialize noise for the current stage
        noise = torch.randn_like(ldr)
        input_t = torch.cat([rgb_ldr, noise], dim=1)

        # estimate residual
        with torch.no_grad():
            residual = generator(input_t).clamp(0, 1)

        # create extended estimate
        est = rgb_ldr + residual

        # prepare input for next stage
        rgb_ldr = torch.clamp(est.clone() / 2, 0, 1)

        executions += 1
        intermediates.append(est.detach().cpu().permute(0, 2, 3, 1)[0].numpy())
        # repeat ...

    # invert the recurrent renormalization to get to absolute HDR scale
    xdr = est * 2 ** (executions - 1)

    # compensate for our initial normalization
    xdr = xdr * brightness_scale

    # resize to original size - undo multiple of 32
    xdr = torch.nn.functional.interpolate(
        xdr,
        size=input_shape,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(xdr.min(), None)

    # soft blend with input
    mask = torch.where(hdr_t > 0.5, hdr_t.clip(0, 1), 0.5)
    mask = (mask - 0.5) / 0.5
    hdr_t = hdr_t * (1 - mask.float()) + (xdr * mask.float())

    # store result
    hdr = to2np(hdr_t)

    output = {
        "hdr": hdr,
        "intermediates": steps,
    }

    return output


def main(
    input_path: str,
    result_path: str,
    save_steps: bool = False,
    inference_size: int = 4096,
    start_id: int = 0,
    end_id: int = None,
    baseline_dataset: bool = False,
    steps: int = 24,
) -> None:
    """
    Recurrent inference for dynamic range extension of LDR images to HDR.

    Args:
        input_path: path to input images (can be a glob pattern)
        result_path: path to save results
        save_steps: if True, save intermediate results
        inference_size: size of the input images for inference (default: 4096)
        start_id: start index for processing images (default: 0)
        end_id: end index for processing images (default: None, process all)
        baseline_dataset: if True, adapt file naming for structure from SingleHDR test data
        steps: maximum number of exposure enhancement stages (default: 24)
    """

    # read input images
    image_list = sorted(glob.glob(input_path + "/*.*"))

    # load model
    extender = load_extension_model(CHECKPOINT_PATH, device)

    os.makedirs(result_path, exist_ok=True)

    for img_id, image_path in tqdm(
        enumerate(image_list[start_id:end_id]), total=len(image_list[start_id:end_id])
    ):
        print(
            f"Processing image {img_id + start_id + 1}/{len(image_list)}: {image_path}"
        )

        # load LDR image
        ldr = load_ldr(image_path)

        # perform recurrent inference
        hdr_result = recurrent_inference(
            generator=extender,
            ldr=ldr,
            steps=steps,
            inference_size=inference_size,
        )

        hdr = hdr_result["hdr"]
        intermediates = hdr_result["intermediates"]

        # save intermediate results
        if save_steps:
            inputs = np.concatenate(intermediates, axis=1)
            tiled = tile_imgs([[inputs]])
            tiled = cv2.cvtColor(tiled.astype(np.float32), cv2.COLOR_RGB2BGR)
            file_name = image_path.split(".")[:-1]
            cv2.imwrite(
                os.path.join(result_path, f"{file_name}_steps.png"),
                (tiled ** (1 / 2.2) * 128).astype(np.uint8),
            )

        # save final HDR result
        file_ending = image_path.split(".")[-1]
        if baseline_dataset:
            file_name = image_path.split("/")[-2] + "/recurrent." + file_ending
        else:
            file_name = os.path.basename(image_path)
        cv2.imwrite(
            os.path.join(
                result_path,
                file_name.replace(file_ending, "exr"),
            ),
            cv2.cvtColor(hdr, cv2.COLOR_RGB2BGR),
        )
    print("Inference complete.")


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Path to input images")
    parser.add_argument("--result_path", type=str, help="Path to save results")
    parser.add_argument(
        "--start_id", type=int, default=0, help="Start index for processing images"
    )
    parser.add_argument(
        "--end_id", type=int, default=None, help="End index for processing images"
    )
    parser.add_argument(
        "--inference_size",
        type=int,
        default=1024,
        help="Size of the input images for inference",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=24,
        help="Maximum number of exposure enhancement stages",
    )
    parser.add_argument(
        "--save_steps",
        action="store_true",
        help="Save intermediate results after inference",
    )
    parser.add_argument(
        "--baseline_dataset",
        action="store_true",
        help="Adapt file naming for structure fro SingleHDR test data",
    )
    args = parser.parse_args()

    # run inference
    main(
        input_path=args.input_path,
        result_path=args.result_path,
        save_steps=args.save_steps,
        inference_size=args.inference_size,
        start_id=args.start_id,
        end_id=args.end_id,
        baseline_dataset=args.baseline_dataset,
        steps=args.steps,
    )
