import argparse
import shutil             # Copy image files.
from pathlib import Path  # Create directories.
import os                 # Count the number of files.
import time

SPLITS = ["train", "val", "test"]


def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser")

    parser.add_argument(
        "--path_to_iam",
        default="iam/words",
        required=False,
        help="Path to a IAM dataset with word-level images. \
              Source: data/words.tgz from https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database. \
              Default: iam/words")

    parser.add_argument(
        "--out_prefix",
        default="iam",
        required=False,
        help="Prefix of the directory to contain output. Default: iam.")

    args = parser.parse_args()
    return args


def create_output_paths(paths_dict):
    """Create output directories if do not exist."""
    for path in paths_dict.values():
        Path(path).mkdir(parents=True, exist_ok=True)


def process_data(args):
    """Create train, val and test partitions for the IAM dataset in the format required in
    https://github.com/clovaai/deep-text-recognition-benchmark#when-you-need-to-train-on-your-own-dataset-or-non-latin-language-datasets.
    This includes two tasks: copy an image from the IAM dataset directory into a partition's directory and
    create a gt.txt with ground truth annotations."""

    # Process images for each partition in a sequence.
    for split in SPLITS:

        print(f"Started processing a {split} partition.")

        # Import a split.
        with open(args.paths_to_splits[split], "rt", encoding="utf-8") as f:
            split_file = f.readlines()

        # Open gt.txt for writing.
        with open(args.paths_to_gt[split], "wt", encoding="utf-8") as f:

            # Iterate over lines in the split file.
            for i, line in enumerate(split_file):

                # Get a fine name and a label.
                f_name_and_threshold, label = line.strip().split(maxsplit=1)
                f_name, _ = f_name_and_threshold.split(
                    ",")  # 2nd item is a threshold, cf https://doi.org/10.1007/s100320200071

                form1, form2, _, _ = f_name.split("-")  # 4 items: form prefix 1, form prefix 2, line, word.
                src_img_path = f"{args.path_to_iam}/{form1}/{form1}-{form2}/{f_name}.png"
                dst_img_path = f"{args.paths_out[split]}/{f_name}.png"

                try:
                    # Copy an image from IAM directory to the IAM-LMDB directory.
                    shutil.copy2(src_img_path, dst_img_path)

                    # Write a line to gt.txt.
                    gt_img_path = "iam" + "/" + f_name + ".png"
                    f.write(f"{gt_img_path}\t{label}\n")
                except:
                    print(f"Error occurred while copying file {src_img_path}.")

                if i > 0 and i % 5000 == 0:
                    print(f"Processed {i} images.")
            print(f"Processed {i} images.")
    return


def validate(args):
    """Compare the number of images with the number of entries in split file
    and with the number of entries in gt.txt."""

    for split in SPLITS:

        # N images in the directory.
        path = args.paths_out[split]
        n_img_files = len([f for f in os.listdir(path)])

        # N files in the split file.
        with open(args.paths_to_splits[split], "rt", encoding="utf-8") as f:
            split_file = f.readlines()
            n_lines_split = len(split_file)

        # N files in gt.txt
        with open(args.paths_to_gt[split], "rt", encoding="utf-8") as f:
            gt_file = f.readlines()
            n_lines_gt = len(gt_file)

        if n_img_files == n_lines_gt and n_img_files == n_lines_split:
            print(f"{split} split has a correct number of images and annotations ({n_img_files}).")
        else:
            raise ValueError(
                f'Mismatch in {split} split: expected number of images: {n_lines_split}. Got {n_img_files} images and {n_lines_gt} lines in gt.txt')


def main():
    # Parse arguments.
    args = parse_args()

    # Path to splits and annotations.
    args.paths_to_splits = {split: f"iam_rwth_partitions/RWTH.iam_word_gt_final.{split}.thresh.txt" for split in SPLITS}

    # Output path for train, validation and test partitions with images and gt,
    # as required in https://github.com/clovaai/deep-text-recognition-benchmark#when-you-need-to-train-on-your-own-dataset-or-non-latin-language-datasets
    args.paths_out = {split: f"{args.out_prefix}_{split}/iam/" for split in SPLITS}
    args.paths_to_gt = {split: f"{args.out_prefix}_{split}/gt.txt" for split in SPLITS}

    # Create output directories.
    create_output_paths(args.paths_out)

    # Process images.
    process_data(args)

    # Validate the number of processed images.
    validate(args)


if __name__ == '__main__':
    tic = time.time()
    main()
    toc = time.time()
    print(f"Elapsed: {(toc - tic) / 60:.2f} min.")
