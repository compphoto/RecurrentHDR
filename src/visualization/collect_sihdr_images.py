import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import argparse
import numpy as np
import cv2
from tqdm import tqdm


def collect_sihdr_images(
    gt_path, baseline_path, baseline_list, output_path, clip_id, file_name
):
    base_name = os.path.splitext(file_name)[0]
    # read file from ground truth path
    gt_image = cv2.imread(os.path.join(gt_path, file_name), cv2.IMREAD_UNCHANGED)
    if gt_image is None:
        print(f"Ground truth image not found: {os.path.join(gt_path, file_name)}")
        return
    gt_image = np.float32(cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB))
    gt_image = gt_image / np.percentile(
        gt_image, clip_id
    )  # set 95th percentile to 1.0, following original SIHDR paper

    # create output directory if not exists
    output_path = os.path.join(output_path, base_name)
    os.makedirs(output_path, exist_ok=True)

    # write gt image
    cv2.imwrite(
        os.path.join(output_path, f"{base_name}_gt.exr"),
        cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR),
    )

    # read and write images from baseline paths
    for baseline in baseline_list:
        # read image
        baseline_image = cv2.imread(
            os.path.join(baseline_path, baseline, f"clip_{clip_id}", file_name),
            cv2.IMREAD_UNCHANGED,
        )
        if baseline_image is None:
            print(
                f"Baseline image not found: {os.path.join(baseline_path, baseline, f'clip_{clip_id}', file_name)}"
            )
            continue
        baseline_image = np.float32(cv2.cvtColor(baseline_image, cv2.COLOR_BGR2RGB))

        # scale match - find linear fit against LDR part of the gt image
        mask = gt_image.max(axis=2) <= 1.0
        a = baseline_image[mask].reshape(-1, 1)
        b = gt_image[mask].reshape(-1, 1)
        x = np.linalg.lstsq(a, b, rcond=None)[0]
        baseline_image = baseline_image * x

        # write baseline image
        cv2.imwrite(
            os.path.join(output_path, f"{base_name}_{baseline}.exr"),
            cv2.cvtColor(baseline_image, cv2.COLOR_RGB2BGR),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect SIHDR images from ground truth and baseline methods."
    )
    parser.add_argument(
        "--gt_path", type=str, required=True, help="Path to ground truth SIHDR images."
    )
    parser.add_argument(
        "--baseline_path",
        type=str,
        required=True,
        help="Path to baseline SIHDR method folders.",
    )
    parser.add_argument(
        "--baseline_list",
        type=str,
        nargs="+",
        required=True,
        help="List of baseline method folder names.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save collected images.",
    )
    parser.add_argument(
        "--clip",
        type=int,
        required=True,
        help="Path to save collected images.",
    )
    parser.add_argument(
        "--file_names",
        type=str,
        nargs="+",
        required=True,
        help="Name of the SIHDR image file to process.",
    )
    args = parser.parse_args()

    gt_path = args.gt_path
    baseline_path = args.baseline_path
    baseline_list = args.baseline_list
    output_path = args.output_path
    clip = args.clip

    for file_name in tqdm(args.file_names):
        collect_sihdr_images(
            gt_path, baseline_path, baseline_list, output_path, clip, file_name
        )
