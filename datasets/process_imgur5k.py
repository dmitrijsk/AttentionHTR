import argparse
import numpy as np
import cv2
import json
import ast  # To convert a string representation of a list into a list.
from pathlib import Path  # To create folders safely.
import time

SPLITS = ['train', 'val', 'test']


def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser")

    parser.add_argument(
        "--path_to_repo",
        default="../IMGUR5K-Handwriting-Dataset/",
        required=False,
        help="Path to a cloned Imgur5K GitHub repo.")

    parser.add_argument(
        "--out",
        default="imgur5k",
        required=False,
        help="Prefix of the directory to contain output.")

    parser.add_argument(
        "--idx_from",
        type=int,
        default=0,
        required=False,
        help="Index of an image to process from. Default: 0 (start from the first image).")

    parser.add_argument(
        "--idx_to",
        type=int,
        default=None,
        required=False,
        help="Index of an image to process to. Default: None (process all images).")

    parser.add_argument(
        "--display",
        action='store_true',
        help="Display each processed image. Default: False.")

    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        required=False,
        help="Print debug info. 0 - important messages only, 1 - more details, 2 - even more details. Default: 0.")

    parser.add_argument(
        "--keep_missing",
        action='store_true',
        help="Export images with missing annotations.")

    args = parser.parse_args()

    if args.verbose >= 1:
        print(f"Parsed arguments: {args}")

    return args


def resize(img, desired_height=32):
    """Resize images before displaying them."""
    h, w = img.shape[0], img.shape[1]
    scale = h / desired_height
    img = cv2.resize(img, (int(w / scale), desired_height))
    return img


def create_output_paths(paths_dict):
    """Create output directories if do not exist."""
    for path in paths_dict.values():
        Path(path).mkdir(parents=True, exist_ok=True)


def change_args(args, n_img):
    """Change args where necessary."""

    # Process *all* images if a user did not specify otherwise.
    if args.idx_to is None:
        args.idx_to = n_img

    return args


def get_image_split(img_name, split_idx):
    """Returns a string 'train', 'val' or 'test' for each image."""
    for split in SPLITS:
        if img_name in split_idx[split]:
            return split

    # If not found in any of the splits.
    raise ValueError(f'Image {img_name} not found in any of the splits.')


def get_train_val_test_indices(args):
    """Return a dict with indices for training, validation and test images."""

    split_idx = {}
    for split in SPLITS:
        split_idx[split] = np.loadtxt(args.paths_to_splits[split],
                                      delimiter="\n",
                                      dtype=np.str_,
                                      encoding="UTF-8")

        if args.verbose >= 1:
            print(f"{split} contains {len(split_idx[split])} indices.")

    return split_idx


def rotate_crop_image(args, img, sub_box):
    """Rotates and crop an image, returns a cropped image."""

    # Elements of the bounding box.
    xc, yc, w, h, a = ast.literal_eval(sub_box)
    # (xc, ys) is the center of the rotated box, and the angle a is in degrees ccw.

    # Get the rotation matrix.
    rotate_matrix = cv2.getRotationMatrix2D(center=(xc, yc), angle=-a, scale=1)

    # Rotate the original image.
    rotated_img = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(img.shape[1], img.shape[0]))

    # Crop the rotated image.
    crop_img = rotated_img[max(0, int(yc - h / 2)):int(yc + h / 2), max(0, int(xc - w / 2)):int(xc + w / 2)]

    if args.display:
        # Display images.
        cv2.imshow('Original', resize(img, 512))
        cv2.imshow('Rotated', resize(rotated_img, 512))
        cv2.imshow("Cropped", resize(crop_img, 64))

        # Press any key to continue.
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return crop_img


def write_gts(args, gts):
    """Write a gt.txt from a dictionary gt as instructed here:
        https://github.com/clovaai/deep-text-recognition-benchmark#when-you-need-to-train-on-your-own-dataset-or-non-latin-language-datasets"""

    for split in SPLITS:
        with open(args.paths_to_gt[split], "wt", encoding="utf-8") as f:
            for k, v in gts[split].items():
                imagepath = k
                label = v
                f.write(f"{imagepath}\t{label}\n")

        if args.verbose >= 1:
            print(f"{args.paths_to_gt[split]} contains {len(gts[split].items())} sub-images.")


def get_image_annotations(args):
    """Import JSON with bounding boxes and annotations."""
    with open(args.path_to_annotations) as json_file:
        data = json.load(json_file)
    return data


def process_data(args, data, split_idx):
    """Export word-level images and corresponding gt annotations."""

    # Separate dict for train / validation / test gt.
    gts = {split: {} for split in SPLITS}

    # Create a list of images to process.
    img_to_process = list(data["index_id"].keys())[args.idx_from:args.idx_to]
    print(f"Started processing {len(img_to_process)} images (of possible {len(data['index_id'])}).")

    for img_name in img_to_process:
        img_path = data["index_id"][img_name]["image_path"]
        sub_images_lst = data["index_to_ann_map"][img_name]

        # Determine the split (train / val / test).
        split = get_image_split(img_name, split_idx)

        if args.verbose >= 1:
            print(
                f"Processing {args.path_to_repo + img_path} into *{split}* set. Contains {len(sub_images_lst)} sub-image(s).")

        if args.verbose >= 2:
            print(f"Sub-images for image {img_name}: {sub_images_lst}")
            print(f"Sub-image data for {img_name}:")

        # Read image.
        img = cv2.imread(args.path_to_repo + img_path)

        # Check if the image file exists.
        if img is None:
            print(f"* {args.path_to_repo + img_path} does not exist.")
            continue

        for sub_image in sub_images_lst:

            # Ground truth label.
            sub_word = data["ann_id"][sub_image]["word"]

            if sub_word == "." and not args.keep_missing:
                # Skip words with missing annotations.
                continue

            # Bounding box.
            sub_box = data["ann_id"][sub_image]["bounding_box"]

            # DO THE WORK (ROTATE & CROP) WITH IMAGES HERE.    
            crop_img = rotate_crop_image(args, img, sub_box)

            # Create a filename for the cropped image.
            ext = img_path.split(".")[-1]
            out_filename = sub_image + "." + ext

            # Export a cropped image.
            cv2.imwrite(args.paths_out[split] + out_filename, crop_img)

            # Save gt info: filepath & label.
            gts[split]["test" + "/" + out_filename] = sub_word

            if args.verbose >= 2:
                print(f"Sub-image: {sub_image}, label: {sub_word}, box: {sub_box}")

        if args.verbose >= 2:
            print()

        if (img_to_process.index(img_name) + 1) % 200 == 0:
            print(f"Update: processed {img_to_process.index(img_name) + 1} images.")

    return gts


def main():
    # Parse arguments.
    args = parse_args()

    args.path_to_annotations = args.path_to_repo + "dataset_info/imgur5k_annotations.json"
    # Structure:
    # "index_id": get "image_path" here.
    # "index_to_ann_map": get a list of all sub-images.
    # "ann_id": get annotations and bounding-boxes for each sub-image.
    args.paths_to_splits = {split: args.path_to_repo + f'dataset_info/{split}_index_ids.lst' for split in SPLITS}
    args.paths_out = {split: f"{args.out}_{split}_{args.idx_from}_{args.idx_to}/test/" for split in SPLITS}
    args.paths_to_gt = {split: f"{args.out}_{split}_{args.idx_from}_{args.idx_to}/gt.txt" for split in SPLITS}

    # Get image annotations.
    data = get_image_annotations(args)

    # Change args where necessary.
    args = change_args(args, n_img=len(data["index_id"]))

    # Create output directories with image indices in the folder names.
    create_output_paths(args.paths_out)

    # Get indices (dict) for training, validation and test sub-images.
    split_idx = get_train_val_test_indices(args)

    # Loop over all images and produce a gt dict.
    gts = process_data(args, data, split_idx)
    print("Finished image processing.")

    # Write gt.txt files.
    write_gts(args, gts)
    print(f"gt.txt exported to {args.paths_to_gt}.")


if __name__ == '__main__':
    tic = time.time()
    main()
    toc = time.time()
    print(f"Elapsed: {(toc - tic) / 60:.2f} min.")
