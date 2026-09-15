import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import argparse
import numpy as np
import cv2
from tqdm import tqdm


def collect_dataset_images(
    input_path, baseline_path, baseline_list, output_path, start_id=0, end_id=None
):
    image_list = os.listdir(input_path)

    for file_name in tqdm(image_list[start_id:end_id]):
        base_name = os.path.splitext(file_name)[0]
        out_dir = os.path.join(output_path, base_name)
        # create output directory if not exists
        os.makedirs(out_dir, exist_ok=True)

        input_image = cv2.imread(
            os.path.join(input_path, file_name), cv2.IMREAD_UNCHANGED
        )
        if input_image is None:
            print(f"Input image not found: {os.path.join(input_path, file_name)}")
            continue

        input_image = np.float32(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        input_image = input_image / 255.0  # normalize to [0, 1]
        input_image = input_image**2.2  # gamma correct to linear space
        # write input image
        cv2.imwrite(
            os.path.join(out_dir, f"{base_name}.png"),
            cv2.cvtColor(np.uint8(input_image ** (1 / 2.2) * 255), cv2.COLOR_RGB2BGR),
        )

        # read and write images from baseline paths
        for baseline in baseline_list:
            # read image
            baseline_image = cv2.imread(
                os.path.join(baseline_path, baseline, base_name + ".exr"),
                cv2.IMREAD_UNCHANGED,
            )
            if baseline_image is None:
                print(
                    f"Baseline image not found: {os.path.join(baseline_path, baseline, file_name)}"
                )
                continue
            baseline_image = np.float32(cv2.cvtColor(baseline_image, cv2.COLOR_BGR2RGB))

            # scale match - find linear fit against LDR part of the gt image
            mask = input_image.max(axis=2) <= 1.0
            a = baseline_image[mask].reshape(-1, 1)
            b = input_image[mask].reshape(-1, 1)
            x = np.linalg.lstsq(a, b, rcond=None)[0]
            baseline_image = baseline_image * x

            # write baseline image
            cv2.imwrite(
                os.path.join(out_dir, f"{base_name}_{baseline}.exr"),
                cv2.cvtColor(baseline_image, cv2.COLOR_RGB2BGR),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect ITW images from input and baseline methods."
    )
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to input images."
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
        "--start_id",
        type=int,
        default=0,
        help="Start ID to process (inclusive).",
    )
    parser.add_argument(
        "--end_id",
        type=int,
        default=None,
        help="End ID to process (exclusive).",
    )
    args = parser.parse_args()

    input_path = args.input_path
    baseline_path = args.baseline_path
    baseline_list = args.baseline_list
    output_path = args.output_path
    start_id = args.start_id
    end_id = args.end_id

    collect_dataset_images(
        input_path, baseline_path, baseline_list, output_path, start_id, end_id
    )
