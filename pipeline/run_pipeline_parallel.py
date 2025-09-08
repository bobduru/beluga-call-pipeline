from dotenv import load_dotenv
from pathlib import Path
import sys
import os
import argparse

# Walk up until we find the project root (folder with the .env)
current_path = Path().resolve()
for parent in [current_path] + list(current_path.parents):
    if (parent / ".env").exists():
        load_dotenv(parent / ".env")
        project_root = os.getenv("PROJECT_ROOT")
        print(project_root)
        sys.path.append(project_root)     
        break


from models.quant_mobilenet import load_mobilenet_v3_quant_from_file
from data_preprocessing.spectrogram.spectrogram_generator import SpectrogramGenerator

import multiprocessing
import time
import argparse
import os
from pipeline import (
    run_multi_file_pipeline_with_prefetching,
    filter_already_processed_files,
    find_wav_files_in_dir,
    merge_multi_file_pipeline_outputs,
    find_continuous_segments
)


def run_pipeline_subset(file_subset, raw_dir, out_dir,  process_index):

    spect_generator = SpectrogramGenerator(
            n_fft=2048,
            hop_length=1024,
            n_mels=64,
            fmin=200,
            sample_rate=192000, 
        )

    return run_multi_file_pipeline_with_prefetching(
        file_subset,
        raw_dir,
        out_dir,
        spect_generator,
        process_index=process_index,
        filter_absences=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-file audio pipeline in parallel.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing WAV files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save pipeline outputs.")
    parser.add_argument("--num_processes", type=int, default=4, help="Number of parallel processes to run.")
    parser.add_argument("--max_files", type=int, default=None, help="Maximum number of WAV files to process.")

    args = parser.parse_args()

    # Load and filter files
    raw_wav_files = find_wav_files_in_dir(args.input_dir, log=True)
    n_files_in_dir = len(raw_wav_files)
    raw_wav_files = filter_already_processed_files(raw_wav_files, args.output_dir)
    print(f"Filtered out {n_files_in_dir-len(raw_wav_files)} files that had already been processed in {args.output_dir}")

    if args.max_files is not None:
        raw_wav_files = raw_wav_files[:args.max_files]

    num_files = len(raw_wav_files)
    if num_files == 0:
        print("No unprocessed WAV files found.")
        exit(0)

    num_processes = min(args.num_processes, num_files)
    chunk_size = num_files // num_processes
    processes = []

    print(f"🛠️ Starting processing of {num_files} files using {num_processes} processes...\n")
    start_time = time.time()

    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_processes - 1 else num_files
        subset = raw_wav_files[start_idx:end_idx]
        print(f"Starting process {i}")
        p = multiprocessing.Process(
            target=run_pipeline_subset,
            args=(subset, args.input_dir, args.output_dir, i)
        )
        processes.append(p)
        p.start()

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("🛑 KeyboardInterrupt received: terminating child processes...")
        for p in processes:
            p.terminate()
            p.join()

    total_time = time.time() - start_time
    print(f"\n✅ Finished processing {num_files} files with {num_processes} processes in {total_time:.2f} seconds.")

    merge_multi_file_pipeline_outputs(args.output_dir)
    find_continuous_segments(args.output_dir)



# python ./pipeline/run_parallel_overlap.py --input_dir E:/2022/ --output_dir ./pipeline/overlap_outputs/BSM_2022 --num_processes 4 --max_files 16
# python ./pipeline/run_parallel_overlap.py --input_dir E:/2018/ --output_dir ./pipeline/overlap_outputs/BSM_2018 --num_processes 4 --max_files 16
# python ./pipeline/run_parallel_overlap.py --input_dir E:/2023/ --output_dir ./pipeline/overlap_outputs/BSM_2023 --num_processes 4 --max_files 16

# Finished processing 16 files with 4 processes in 130.83 seconds.
# Finished processing 16 files with 8 processes in 274.67 seconds.

