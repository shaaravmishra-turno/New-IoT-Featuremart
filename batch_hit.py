"""
Batch processor for IOT Feature Mart.

Reads VINs from vins_copy.txt, processes them in batches of BATCH_SIZE
through final_iot.py, and consolidates all results into a single CSV.

Features:
  - Resumable: tracks processed VINs in processed_vins.txt; skip on restart.
  - Interruptible: Ctrl+C stops gracefully after current batch attempt.
  - Credential rotation: change .env between runs; re-read on each batch.
  - Consolidated output: appends each batch's results to consolidated_features.csv.

Usage:
  python batch_hit.py                    # defaults: batch_size=500, days=7
  python batch_hit.py 300                # custom batch size, days=7
  python batch_hit.py 300 14             # custom batch size and days
"""

import os
import sys
import glob
import time
import subprocess
from datetime import datetime

# --------------- Configuration ---------------
BATCH_SIZE = 500
DAYS = 7
VINS_SOURCE_FILE = "vins_copy.txt"
VINS_FILE = "vins.txt"
PROCESSED_VINS_FILE = "processed_vins.txt"
CONSOLIDATED_OUTPUT_FILE = "consolidated_features.csv"
SKIPPED_VINS_FILE = "skipped_vins.txt"
BATCH_OUTPUT_FILE = "_batch_output_temp.csv"
# ----------------------------------------------


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [batch_hit] {msg}")


def load_vins(filepath):
    """Load VINs from a file, one per line, stripping whitespace and blanks."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r") as f:
        return [line.strip() for line in f if line.strip()]


def save_vins(filepath, vins):
    """Write VINs to a file, one per line."""
    with open(filepath, "w") as f:
        for vin in vins:
            f.write(vin + "\n")


def append_processed_vins(vins):
    """Append successfully-attempted VINs to the tracking file."""
    with open(PROCESSED_VINS_FILE, "a") as f:
        for vin in vins:
            f.write(vin + "\n")


def append_to_consolidated(batch_output_file, consolidated_has_header):
    """
    Append batch output CSV to the consolidated file.
    Returns True if data was written, False otherwise.
    """
    if not os.path.exists(batch_output_file):
        log(f"WARNING: Batch output file {batch_output_file} not found!")
        return False

    with open(batch_output_file, "r") as src:
        lines = src.readlines()

    if not lines or (len(lines) == 1 and lines[0].strip() == ""):
        log(f"WARNING: Batch output file {batch_output_file} is empty!")
        return False

    with open(CONSOLIDATED_OUTPUT_FILE, "a") as dst:
        if consolidated_has_header:
            # Skip header row; append only data rows
            dst.writelines(lines[1:])
        else:
            # First write: include header + data
            dst.writelines(lines)

    return True


def main():
    global BATCH_SIZE, DAYS

    # Parse optional CLI arguments
    if len(sys.argv) > 1:
        BATCH_SIZE = int(sys.argv[1])
    if len(sys.argv) > 2:
        DAYS = int(sys.argv[2])

    log("=" * 70)
    log("BATCH IOT FEATURE PROCESSING — START")
    log(f"Batch size : {BATCH_SIZE}")
    log(f"Days       : {DAYS}")
    log(f"Source     : {VINS_SOURCE_FILE}")
    log(f"Output     : {CONSOLIDATED_OUTPUT_FILE}")
    log(f"Tracker    : {PROCESSED_VINS_FILE}")
    log("=" * 70)

    # ---- Load all VINs from source ----
    all_vins = load_vins(VINS_SOURCE_FILE)
    log(f"Total VINs in {VINS_SOURCE_FILE}: {len(all_vins)}")
    if not all_vins:
        log(f"No VINs found in {VINS_SOURCE_FILE}. Exiting.")
        return

    # ---- Load already-processed VINs (for resume) ----
    processed_vins = set(load_vins(PROCESSED_VINS_FILE))
    log(f"Already processed VINs (from {PROCESSED_VINS_FILE}): {len(processed_vins)}")

    # ---- Load previously-skipped VINs ----
    skipped_vins = set(load_vins(SKIPPED_VINS_FILE))
    log(f"Previously skipped VINs (from {SKIPPED_VINS_FILE}): {len(skipped_vins)}")

    # ---- Filter out processed + skipped ----
    already_done = processed_vins | skipped_vins
    remaining_vins = [v for v in all_vins if v not in already_done]
    log(f"Remaining VINs to process: {len(remaining_vins)}")

    if not remaining_vins:
        log("All VINs already processed or skipped. Nothing to do.")
        return

    # ---- Split into batches ----
    batches = [
        remaining_vins[i : i + BATCH_SIZE]
        for i in range(0, len(remaining_vins), BATCH_SIZE)
    ]
    total_batches = len(batches)
    log(f"Batches to run: {total_batches}")

    # Check whether consolidated file already has a header row
    consolidated_has_header = (
        os.path.exists(CONSOLIDATED_OUTPUT_FILE)
        and os.path.getsize(CONSOLIDATED_OUTPUT_FILE) > 0
    )

    # ---- Process each batch ----
    for batch_idx, batch_vins in enumerate(batches, 1):
        log("-" * 70)
        log(f"BATCH {batch_idx}/{total_batches}  |  {len(batch_vins)} VINs")
        log(f"First VIN: {batch_vins[0]}  |  Last VIN: {batch_vins[-1]}")
        log("-" * 70)

        # 1. Write current batch to vins.txt
        save_vins(VINS_FILE, batch_vins)
        log(f"Wrote {len(batch_vins)} VINs to {VINS_FILE}")

        # 2. Clean up temp output file from any prior run
        if os.path.exists(BATCH_OUTPUT_FILE):
            os.remove(BATCH_OUTPUT_FILE)

        # 3. Run final_iot.py as a subprocess
        #    Args: days, output_file
        cmd = [sys.executable, "final_iot.py", str(DAYS), BATCH_OUTPUT_FILE]
        log(f"Running: {' '.join(cmd)}")
        batch_start = time.time()

        try:
            result = subprocess.run(cmd)

            batch_elapsed = time.time() - batch_start
            log(f"final_iot.py finished (exit code {result.returncode}) in {batch_elapsed:.1f}s")

            if result.returncode != 0:
                log(f"WARNING: final_iot.py exited with non-zero code {result.returncode}")
                log("Skipping append for this batch but marking VINs as attempted.")

        except KeyboardInterrupt:
            log("")
            log("*" * 70)
            log("INTERRUPTED by user (Ctrl+C)")
            log(f"Batch {batch_idx} was NOT completed.")
            log(f"Processed VINs so far are saved in: {PROCESSED_VINS_FILE}")
            log(f"Consolidated output so far is in  : {CONSOLIDATED_OUTPUT_FILE}")
            log("")
            log("To RESUME, simply run:")
            log(f"    python batch_hit.py {BATCH_SIZE} {DAYS}")
            log("")
            log("You can change credentials in .env before resuming.")
            log("Already-processed VINs will be skipped automatically.")
            log("*" * 70)
            return

        except Exception as e:
            log(f"ERROR running final_iot.py: {e}")
            log(f"To resume: python batch_hit.py {BATCH_SIZE} {DAYS}")
            return

        # 4. Append batch results to consolidated file
        if os.path.exists(BATCH_OUTPUT_FILE):
            if append_to_consolidated(BATCH_OUTPUT_FILE, consolidated_has_header):
                consolidated_has_header = True
                log(f"Appended batch results to {CONSOLIDATED_OUTPUT_FILE}")

                # Get row count of the batch output (minus header)
                with open(BATCH_OUTPUT_FILE, "r") as f:
                    row_count = sum(1 for _ in f) - 1
                log(f"Batch produced {row_count} feature rows")
            else:
                log("No data appended (batch may have produced empty output)")

            # Clean up temp file
            try:
                os.remove(BATCH_OUTPUT_FILE)
            except Exception:
                pass
        else:
            log(f"No output file found — VINs in this batch may have had no IOT data")

        # 5. Mark all batch VINs as processed (so they are skipped on resume)
        append_processed_vins(batch_vins)
        processed_vins.update(batch_vins)
        log(f"Marked {len(batch_vins)} VINs as processed")

        # 6. Progress summary
        total_done = len(processed_vins) + len(skipped_vins)
        total_left = len(all_vins) - total_done
        log(f"Progress: {total_done}/{len(all_vins)} done  |  {total_left} remaining")
        log("=" * 70)

    # ---- All batches complete ----
    log("")
    log("=" * 70)
    log("ALL BATCHES COMPLETED SUCCESSFULLY")
    log(f"Consolidated output : {CONSOLIDATED_OUTPUT_FILE}")
    log(f"Processed VINs log  : {PROCESSED_VINS_FILE}")
    log(f"Skipped VINs log    : {SKIPPED_VINS_FILE}")
    log(f"Total processed     : {len(processed_vins)}")
    log(f"Total skipped       : {len(skipped_vins)}")
    log("=" * 70)


if __name__ == "__main__":
    main()
