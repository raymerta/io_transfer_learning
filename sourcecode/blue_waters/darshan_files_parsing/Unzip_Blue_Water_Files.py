import gzip
from math import ceil
from pathlib import Path
import shutil
import tqdm
from tqdm.contrib.concurrent import process_map

dataset_dir = Path("/work/thes1067/data/blue_waters_dataset")


def unzip_file(gzipped_file):
    unpacked_file = gzipped_file.with_suffix('')

    with gzip.open(gzipped_file, 'rb') as f_in:
        with open(unpacked_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    unpacked_file.unlink()


if __name__ == "__main__":

    dirs_with_darshan_files = [d for d in dataset_dir.iterdir() if d.is_dir()]

    darshan_file_dir_progress_bar = tqdm.tqdm(dirs_with_darshan_files, position=1)

    for darshan_file_dir in darshan_file_dir_progress_bar:
        darshan_file_dir_progress_bar.set_description(f"Processing {darshan_file_dir.name}")
        darshan_files = list(darshan_file_dir.glob("*.gz")) 

        if len(darshan_files) == 0:
            pass

        chunksize = 1
        if len(darshan_files) > 160:
            chunksize = max(ceil(len(darshan_files) / 16), 1)

        process_map(unzip_file, darshan_files, max_workers = 1, chunksize=chunksize, position=0)