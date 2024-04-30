# Merge individual parsed CSVs into one file

import pandas as pd
from math import ceil
from pathlib import Path
import sys

from typing import Set

from multiprocessing import Pool

NUM_WORKERS = 48

def load_csv_report(path_to_csv_file: Path):
    try:
        return pd.read_csv(path_to_csv_file)

    except pd.errors.EmptyDataError:
        print(f"Error: Empty CSV file{path_to_csv_file}")

def find_files_in_current_dir(current_dir_path: Path):

    posix_csv_file_paths = set(current_dir_path.glob("*_posix.csv"))
    print(f"Found {len(posix_csv_file_paths)} POSIX CSV files in {current_dir_path.name}")

    return posix_csv_file_paths


def find_all_files_recursively(dataset_dir: Path):
    csv_files_to_process: Set[Path] = set()
    dirs_to_process = [d for d in dataset_dir.iterdir() if d.is_dir()]

    num_workers = min(len(dirs_to_process), NUM_WORKERS)
    chunksize = 1
    if len(dirs_to_process) > num_workers:
        chunksize = max(ceil(len(dirs_to_process) / num_workers * 10), 1)

    with Pool(processes=num_workers, maxtasksperchild=1000) as pool:
        for result in pool.imap_unordered(
            find_files_in_current_dir, dirs_to_process, chunksize=chunksize
        ):
            files_to_process_in_dir = result

            csv_files_to_process.update(files_to_process_in_dir)

    return csv_files_to_process




if __name__ == "__main__":
    dataset_dir = Path(sys.argv[1])

    paths_to_posix_csv_files = find_all_files_recursively(
        dataset_dir
    )

    paths_to_posix_csv_files_len = len(paths_to_posix_csv_files)
    print(f"{paths_to_posix_csv_files_len} files found across all folders")

    if paths_to_posix_csv_files_len == 0:
        print("No files left to parse. Exiting...")
        sys.exit()

    chunksize = 100
    total_processed_files = 0
    processed_files = []

    with Pool(processes=NUM_WORKERS, maxtasksperchild=1000) as pool:
        for result in pool.imap_unordered(load_csv_report, paths_to_posix_csv_files, chunksize=chunksize):
            processed_files.append(result)

            total_processed_files += 1
            if total_processed_files % 100 == 0:
                print(
                    f"Processed file {total_processed_files} out of {paths_to_posix_csv_files_len}"
                )

    combined_df = pd.concat(processed_files, ignore_index=True, copy=False)
    combined_df.to_csv(dataset_dir.joinpath("blue_waters_posix.csv"), index=False)

    print("All done!")
    print(
        f"Processed files: {total_processed_files} ({(float(total_processed_files) / float(paths_to_posix_csv_files_len)) * 100}%)"
    )
