import os
import sys

sys.path.append("../src")
import cv2
import numpy as np
import argparse
from tqdm import tqdm

from src.utils import load_raw


def preprocess_raw_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in tqdm(os.listdir(input_dir)):
        raw_path = os.path.join(input_dir, filename)
        raw = load_raw(raw_path)

        ldr = raw.clip(0, 1)

        # Save the processed image
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".png")
        ldr = (np.clip(ldr * 255.0, 0, 255)).astype(np.uint8)
        ldr = cv2.cvtColor(ldr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, ldr)

        output_path = os.path.join(
            output_dir, os.path.splitext(filename)[0] + "_step1.hdr"
        )
        step = raw.clip(0, 2)
        step = cv2.cvtColor(step, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, step)

        output_path = os.path.join(
            output_dir, os.path.splitext(filename)[0] + "_gt.hdr"
        )
        raw = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, raw)

        # print(f"Processed and saved: {output_path}")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(
        description="Preprocess raw images to LDR format."
    )
    argument_parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing raw images."
    )
    argument_parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save processed images.",
    )
    args = argument_parser.parse_args()
    input_directory = args.input_dir
    output_directory = args.output_dir
    preprocess_raw_images(input_directory, output_directory)
