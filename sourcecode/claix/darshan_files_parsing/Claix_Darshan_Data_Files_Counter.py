# coding: utf-8
import sys

# # Parse POSIX binary Darshan files from CLAIX dataset

import pandas as pd
from pathlib import Path

def load_or_create_processed_files_log(log_path: Path) -> pd.DataFrame:
    if log_path.exists:
        processed_files_log = pd.read_csv(log_path)
    else:
        processed_files_log = pd.DataFrame(columns=["File", "Processed", "Comments"])

    return processed_files_log


def get_already_processed_files(darshan_file_dir):
    processed_files_number = 0
    processed_files = set()
    for file in darshan_file_dir.glob("*_posix.csv"):
        original_file_name = Path(str(file).replace("_posix.csv", ".darshan"))

        processed_files.add(original_file_name)


        original_file_name = original_file_name.with_suffix(".gz")
        processed_files.add(original_file_name)
        processed_files_number += 1

    return processed_files, processed_files_number


if __name__ == "__main__":
    dataset_dir = Path(sys.argv[1])

    dirs_with_darshan_files = [d for d in dataset_dir.iterdir() if d.is_dir()]

    total_files = 0
    unprocessed_files_count = 0
    # Go through both gzipped and unpacked Darshan files in a directory
    current_dir_number = 0
    number_of_dirs = len(dirs_with_darshan_files)
    for darshan_file_dir in dirs_with_darshan_files:

        print(
            f"Processing directory {darshan_file_dir.name} ({current_dir_number+1}/{number_of_dirs})"
        )

        current_dir_number += 1

        darshan_files = set(darshan_file_dir.glob("*.darshan*"))
        total_files += len(darshan_files)
        print(f"Found {len(darshan_files)} Darshan files")

        processed_files, processed_files_number = get_already_processed_files(darshan_file_dir)

        print(f"{processed_files_number} already processed before")

        darshan_files = darshan_files - processed_files
        del processed_files

        unprocessed_files_count += len(darshan_files)
        print(f"{len(darshan_files)} Darshan files left to process in this folder")

    print("All done!")
    print(
        f"Unprocessed files: {unprocessed_files_count} ({(float(unprocessed_files_count) / float(total_files)) * 100}%)"
    )
    print(f"Total Darshan files found: {total_files}")
