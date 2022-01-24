import urllib.request     # Read from remote text files.
import shutil             # Copy image files.
from pathlib import Path  # Create directories.
import os                 # Count the number of files.

SPLITS = ["train", "val", "test"]

# Path to the IAM dataset on the word level.
# Source: data/words.tgz from https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database
IMG_ROOT = "iam/words"

# Output path for train, validation and test partitions with images and gt.
# This format is as required in https://github.com/clovaai/deep-text-recognition-benchmark#when-you-need-to-train-on-your-own-dataset-or-non-latin-language-datasets
LMDB_IMG_PATHS = {split: f"iam-lmdb/iam-{split}/test" for split in SPLITS}
LMDB_GT_PATHS = {split: f"iam-lmdb/iam-{split}/gt.txt" for split in SPLITS}

# Use the same splits as in https://doi.org/10.1007/978-3-030-12939-2_32 for a fair comparison.
SPLIT_URLS = {
    "train": "https://raw.githubusercontent.com/omni-us/research-seq2seq-HTR/master/RWTH_partition/RWTH.iam_word_gt_final.train.thresh",
    "val": "https://raw.githubusercontent.com/omni-us/research-seq2seq-HTR/master/RWTH_partition/RWTH.iam_word_gt_final.valid.thresh",
    "test": "https://raw.githubusercontent.com/omni-us/research-seq2seq-HTR/master/RWTH_partition/RWTH.iam_word_gt_final.test.thresh"
}

# Create output directories.
for path in LMDB_IMG_PATHS.values():
    Path(path).mkdir(parents=True, exist_ok=True)

# Counter of entries written to gt.txt.
# This will be used to validate the result.
gt_line_counter = {split: 0 for split in SPLITS}

# Process images for each partition in a sequence.
for split in SPLITS:

    # GitHub text file with splits.
    response = urllib.request.urlopen(SPLIT_URLS[split])
    split_file = response.readlines()

    # Open gt.txt for writing.
    with open(LMDB_GT_PATHS[split], "wt", encoding="utf-8") as f:

        # Iterate over lines in the split file.
        for line in split_file:

            # Get a fine name and a label.
            decoded_line = line.decode("utf-8")
            f_name_and_threshold, label = decoded_line.strip().split(maxsplit=1)
            f_name, _ = f_name_and_threshold.split(
                ",")  # 2nd item is a threshold, cf https://doi.org/10.1007/s100320200071

            form1, form2, _, _ = f_name.split("-")  # 4 items: form prefix 1, form prefix 2, line, word.
            src_img_path = f"{IMG_ROOT}/{form1}/{form1}-{form2}/{f_name}.png"
            dst_img_path = f"{LMDB_IMG_PATHS[split]}/{f_name}.png"

            try:
                # Copy an image from IAM directory to the IAM-LMDB directory.
                shutil.copy2(src_img_path, dst_img_path)

                # Write a line to gt.txt.
                gt_img_path = "test" + "/" + f_name + ".png"
                f.write(f"{gt_img_path}\t{label}\n")
                gt_line_counter[split] += 1
            except:
                print(f"Error occurred while copying file {src_img_path}.")


# Check the number of files.
for split in SPLITS:
    # N files in the directory.
    path = LMDB_IMG_PATHS[split]
    n_img_files = len([f for f in os.listdir(path)])
    # N files in gt.txt
    n_lines_gt = gt_line_counter[split]
    # N files in GitHub split.
    n_lines_split = len(split_file)

    if n_img_files == n_lines_gt and n_img_files == n_lines_split:
        print(f"{split} split has a correct number of images and annotations ({n_img_files}).")
    else:
        raise ValueError(
            f'Mismatch in {split} split: expected number of images: {n_lines_split}. Got {n_img_files} images and {n_lines_gt} lines in gt.txt')
