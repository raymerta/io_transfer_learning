# coding: utf-8

# # Parse POSIX binary Darshan files from Blue Waters dataset

# ### What data to save?

# The summary of PyDarshan provides a lot of different information. However, the **'agg_ioops' node**:
# ```python
# report.summary['agg_ioops']['POSIX']
# ```
# will contain the **same data for POSIX as in the 'counters' dataset**:
# ```python
# records['POSIX'].to_df()['counters']
# ```
#

# The only things missing now are **times** and **bandwidth**.
# - **times** are stored in the 'f_counters' dataset:
#     ```python
#     records['POSIX'].to_df()['fcounters']
#     ```
# **But the times are multiple, and we need a single record. Aggregate:**
# - POSIX_F_READ_TIME, POSIX_F_WRITE_TIME, POSIX_F_META_TIME - _**sum**_
# - POSIX_F_MAX_READ_TIME, POSIX_F_MAX_WRITE_TIME - _**max**_
# - POSIX_F_FASTEST_RANK_TIME - min?
# - POSIX_F_SLOWEST_RANK_TIME - max?
# - POSIX_F_VARIANCE_RANK_TIME - max?
# - POSIX_F_VARIANCE_RANK_BYTES - max?
# - POSIX_F_OPEN_START_TIMESTAMP 	POSIX_F_READ_START_TIMESTAMP 	POSIX_F_WRITE_START_TIMESTAMP 	POSIX_F_CLOSE_START_TIMESTAMP 	POSIX_F_OPEN_END_TIMESTAMP 	POSIX_F_READ_END_TIMESTAMP 	POSIX_F_WRITE_END_TIMESTAMP 	POSIX_F_CLOSE_END_TIMESTAMP (**timestamps**) - ?

# - **bandwidth** needs to be manually calculated.


# ## Calculate the bandiwdth

# There can be several records for the same rank - this means the node was doing I/O with multiple files. In this case, first we need to group them.

# The formula for bandwidth is **slowest rank bandiwdth** taken from the Darshan code - we look for the **longest times** and the **total number of bytes**.

# ```C
# pdata->agg_perf_by_slowest = ((double)pdata->total_bytes / 1048576.0) /
#                                      (pdata->slowest_rank_time +
#                                       pdata->shared_time_by_slowest);
# ```

# Where the **slowest_rank_time** is assumed to be POSIX_F_READ_TIME + POSIX_F_WRITE_TIME based on the information from Jay, the creator of Darshan.

from contextlib import closing
import sqlite3
import pandas as pd
import darshan
import gzip
from math import ceil
import shutil
import copy
from pathlib import Path
import sys
from typing import Set
import traceback
import os

from multiprocessing import Pool

# Needed to access automatic summary via summarize()
darshan.enable_experimental()


NUM_WORKERS = 48

def unzip_file(gzipped_file: Path):
    unpacked_file = gzipped_file.with_suffix("")


    with gzip.open(gzipped_file, "rb") as f_in:
        with open(unpacked_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    return unpacked_file


def parse_darshan_file(input_file_path: Path):

    result = {"File": input_file_path, "Processed": False, "Comments": ""}

    output_file_name = Path(
        str(input_file_path.parent.joinpath(input_file_path.stem)) + "_posix.csv"
    )


    if input_file_path.suffix == ".gz":
        try:
            darshan_file_path = unzip_file(input_file_path)

        except EOFError as e:
            result["Comments"] = e.args[0]
            return result

    else:
        darshan_file_path = input_file_path

    # ### Load binary file
    try:
        # By default, PyDarshan loads records for ALL modules at init time.
        # This takes up a LOT of memory (up to 3.5 GiB) and wastes time.
        # We are only interested in reports with POSIX records
        # -> check if POSIX is present, then load records ONLY for this module
        report = darshan.DarshanReport(str(darshan_file_path), read_all=False)

        if "POSIX" not in report.modules:
            result["Comments"] = "No POSIX records found."
            return result

        report.mod_read_all_records("POSIX")
        report.summarize()

        # ### Read performance counters
        posix_counters = report.records["POSIX"].to_df()["counters"]
        posix_f_counters = report.records["POSIX"].to_df()["fcounters"]

        # ### Calculate the time
        # #### Group the time records by rank
        grouped_f_counters = posix_f_counters.groupby(by="rank")
        f_counters_agg = grouped_f_counters.agg(
            {
                "POSIX_F_READ_TIME": "max",
                "POSIX_F_WRITE_TIME": "max",
                "POSIX_F_META_TIME": "max",
            }
        )

        f_counters_agg["POSIX_TOTAL_TIME"] = (
            f_counters_agg["POSIX_F_READ_TIME"]
            + f_counters_agg["POSIX_F_WRITE_TIME"]
            + f_counters_agg["POSIX_F_META_TIME"]
        )

        slowest_rank = f_counters_agg["POSIX_TOTAL_TIME"].idxmax()
        slowest_rank_row = f_counters_agg.loc[[slowest_rank]]

        # Need to extract the float value from the resulting Series with 1 record
        slowest_rank_time = slowest_rank_row.loc[slowest_rank]["POSIX_TOTAL_TIME"]

        # ### Calculate the bytes read and written
        total_bytes = posix_counters.agg(
            {"POSIX_BYTES_READ": "sum", "POSIX_BYTES_WRITTEN": "sum"}
        )

        # ### Calculate the final bandwidth
        bandwidth = (
            (total_bytes["POSIX_BYTES_READ"] + total_bytes["POSIX_BYTES_WRITTEN"])
            / 1024
            / 1024
        ) / slowest_rank_time

        # ## Assemble everything into a final CSV for saving
        # ### POSIX I/O Ops
        # Convert POSIX I/O ops data from the summary into a DataFrame
        report_posix = pd.DataFrame(report.summary["agg_ioops"]["POSIX"], index=[0])

        # ### POSIX I/O Histogram
        # Convert POSIX I/O histogram data from the summary into a DataFrame
        report_posix_hist = pd.DataFrame(
            report.summary["agg_iohist"]["POSIX"], index=[0]
        )

        # ### Job Metadata from Darshan
        # Do a deep copy to avoid changing the metadata of PyDarshan report
        report_metadata = copy.copy(report.metadata["job"])

        # Change start_time and end_time from timestamps to datetime (already parsed by PyDarshan)
        report_metadata["start_time"] = report.start_time
        report_metadata["end_time"] = report.end_time

        # Flatten the resulting dictionary
        report_metadata["lib_ver"] = report_metadata["metadata"]["lib_ver"]
        report_metadata["hints"] = report_metadata["metadata"]["h"]
        report_metadata.pop("metadata")

        # Add it to the CSV summary
        report_metadata_df = pd.DataFrame(report_metadata, index=[0])

        # ### POSIX Times & Variances
        report_read_write_meta_time = slowest_rank_row.reset_index()

        report_fcounters = (
            posix_f_counters.agg(
                {
                    "POSIX_F_MAX_READ_TIME": "max",
                    "POSIX_F_MAX_WRITE_TIME": "max",
                    "POSIX_F_FASTEST_RANK_TIME": "min",
                    "POSIX_F_SLOWEST_RANK_TIME": "max",
                    "POSIX_F_VARIANCE_RANK_TIME": "max",
                    "POSIX_F_VARIANCE_RANK_BYTES": "max",
                }
            )
            .to_frame()
            .transpose()
        )

        # Merge the data into 1 output CSV
        report_output = pd.concat(
            [
                report_posix,
                report_posix_hist,
                report_read_write_meta_time,
                report_fcounters,
                report_metadata_df,
            ],
            axis=1,
        )

        # ### POSIX Bandwidth
        report_output["bandwidth"] = bandwidth

        # Save the resulting CSV file

        report_output.to_csv(output_file_name, index=False)

        # Parsing succesful, return corresponding message
        result["Processed"] = True

    # If PyDarshan fails to read the file e.g. due to old version (pre 3.2), log it
    except RuntimeError as error:
        result["Comments"] = error.args[0]

    # Remove the unpacked file to reuse space
    finally:
        if input_file_path.suffix == ".gz":
            darshan_file_path.unlink()

        return result


def open_or_create_log(dataset_dir: Path):
    path_to_log_db = dataset_dir.joinpath("processed_files_posix.db")
    log_connection = sqlite3.connect(path_to_log_db)
    log_cursor = log_connection.cursor()

    # Setup log table (if the DB is new)
    log_cursor.execute(
        "CREATE TABLE IF NOT EXISTS log ("
        + "File TEXT NOT NULL,"
        + "Processed BOOLEAN NOT NULL,"
        + "Comments TEXT);"
    )

    return log_connection, log_cursor


def read_processed_files_from_log(log_cursor: sqlite3.Cursor):
    files_from_log_db = log_cursor.execute(
        "SELECT File from log;"
    ).fetchall()

    return set([Path(file) for row in files_from_log_db for file in row])


def add_file_to_log(
    result, log_cursor: sqlite3.Cursor, log_connection: sqlite3.Connection
):
    log_cursor.execute(
        "INSERT INTO log VALUES (?, ?, ?)",
        (str(result["File"]), result["Processed"], result["Comments"]),
    )
    log_connection.commit()



def get_already_processed_files(darshan_file_dir: Path):

    processed_files: Set[Path] = set()
    processed_files_number = 0
    for file in darshan_file_dir.glob("*_posix.csv"):

        # Darshan file can be either be already unpacked or still gzipped, need to check for both
        original_file_name = Path(str(file).replace("_posix.csv", ".darshan"))
        gzipped_file_name = original_file_name.with_suffix(".gz")

        if (
            original_file_name in processed_files
            or gzipped_file_name in processed_files
        ):
            continue

        # Whether it's gzipped or not, it still is a single file
        processed_files_number += 1

        processed_files.add(original_file_name)
        processed_files.add(gzipped_file_name)

    return processed_files, processed_files_number


def find_darshan_files_in_current_dir(current_dir_path: Path):

    darshan_files = set(current_dir_path.glob("*.darshan*"))
    processed_files, processed_files_number = get_already_processed_files(
        current_dir_path
    )

    darshan_files = darshan_files - processed_files
    del processed_files

    return darshan_files, len(darshan_files), processed_files_number


def find_all_darshan_files_recursively(dataset_dir: Path):
    darshan_files_to_process: Set[Path] = set()
    dirs_to_process = [d for d in dataset_dir.iterdir() if d.is_dir()]

    total_files = 0
    previously_processed_files = 0

    num_workers = min(len(dirs_to_process), NUM_WORKERS)
    chunksize = 1
    if len(dirs_to_process) > num_workers:
        chunksize = max(ceil(len(dirs_to_process) / num_workers * 10), 1)

    with Pool(processes=num_workers, maxtasksperchild=1000) as pool:
        for result in pool.imap_unordered(
            find_darshan_files_in_current_dir, dirs_to_process, chunksize=chunksize
        ):
            darshan_files_to_process_in_dir, total_darshan_files_in_dir, processed_darshan_files_in_dir_num = result

            darshan_files_to_process.update(darshan_files_to_process_in_dir)
            total_files += total_darshan_files_in_dir
            previously_processed_files += processed_darshan_files_in_dir_num

    return darshan_files_to_process, total_files, previously_processed_files


if __name__ == "__main__":
    dataset_dir = Path(sys.argv[1])

    darshan_files, total_files, previously_processed_files = find_all_darshan_files_recursively(
        dataset_dir
    )

    print(f"{total_files} files found across all folders")
    print(f"Another {previously_processed_files} were already processed before")

    log_connection, log_cursor = open_or_create_log(dataset_dir)
    files_in_log = read_processed_files_from_log(log_cursor)

    print(f"{len(files_in_log)} in log. Removing them...")

    darshan_files = darshan_files - files_in_log
    del files_in_log

    darshan_files_num = len(darshan_files)
    print(f"{darshan_files_num} Darshan files left to process")

    # All files have been already processed
    if len(darshan_files) == 0:
        print("No files left to parse. Exiting...")
        sys.exit()

    chunksize = 100
    total_processed_files = 0

    with closing(log_connection):
        with closing(log_cursor):

            
            sys.stderr = os.devnull

            try:
                with Pool(processes=NUM_WORKERS) as pool: #, maxtasksperchild=1000
                    for result in pool.imap_unordered(parse_darshan_file, darshan_files, chunksize=chunksize):
                        add_file_to_log(result, log_cursor, log_connection)

            except Exception as e:
                traceback.print_exception(file=sys.stdout)
                traceback.print_stack(file=sys.stdout)


    print("All done!")