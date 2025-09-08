# Real-Time Acoustic Monitoring: Lightweight Beluga Call Classification Across Habitats

This project presents a deep learning pipeline for real-time classification of beluga whale vocalizations using Passive Acoustic Monitoring (PAM) data from the St. Lawrence Estuary. The repository contains:

- **Training experiments** for model optimization and cross-site generalization
- **Lightweight model** (320 KB) optimized for deployment, achieving state-of-the-art performance in classifying overlapping beluga calls (F1-score: 0.93)
- **Production pipeline** capable of processing single WAV files for quick testing or months of PAM data in parallel


All technical details are available in the accompanying `paper.pdf`.

---

## Installation

To install the necessary dependencies for this project, follow these steps:

1. Ensure Python is properly installed on your machine.

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Setup

After installing the dependencies, create a `.env` file in the root directory and set:

```
PROJECT_ROOT=/your/root/dir/beluga-call-pipeline
```

This is used in each `.ipynb` file to ensure imports work correctly across the project, using this snippet:

```python
from dotenv import load_dotenv
from pathlib import Path
import sys
import os

# Walk up until we find the project root (folder with the .env)
current_path = Path().resolve()
for parent in [current_path] + list(current_path.parents):
    if (parent / ".env").exists():
        load_dotenv(parent / ".env")
        project_root = os.getenv("PROJECT_ROOT")
        print(project_root)
        sys.path.append(project_root)
        break

%load_ext autoreload
%autoreload 2
```

---

## Data

The data used in this project to train the models is owned by the LISSE Lab in Quebec. To get access to it, please contact [projetbruit2020@gmail.com](mailto:projetbruit2020@gmail.com).

---

## Project Structure

The folders and files should be mostly self-explanatory. Most of the core logic and functions are in `.py` files, while we've kept `.ipynb` notebooks to run the main experiments and the pipeline, allowing for quick testing and familiarization with the code.

---

## Running the Pipeline

The final model is only 320 KB, so the model weights are included directly in the repository.

To get started with using the pipeline, go to `pipeline/run_pipeline.ipynb`. This notebook demonstrates the basic functionality of the pipeline and includes visualization tools that can be useful for evaluating the pipeline's performance on your files. Thanks to the small model size and other optimizations, depending on your machine, we've been able to process 2-hour WAV files in 20-60 seconds.

**Note:** The spectrogram generation uses the sensitivity of the hydrophone to get exact dB values in the pipeline for noise calculation. If you're working with audio files recorded by a hydrophone model that isn't listed in `./spectrogram/hydrophones_sensitivity.json`, you need to manually add its sensitivity value.

To do so:

1. Extract the hydrophone model number from the filename — it's the number before the first period (e.g., `6872` in `6872.220716021453.wav`).
2. Go to [Ocean Instruments' calibration tool](https://oceaninstruments.azurewebsites.net/) and search for your hydrophone model.
3. Look for the **"End-to-End Calibration – High Gain"** value.
4. Add this value to the JSON file **with a negative sign**, like this:
   ```json
   {
       "6872": -177.6
   }
   ```
   
If this is not done, the pipeline will still work with a default sensitivity of -170, but the resulting dB values won't be exact.

## Running the Pipeline on Long Periods of Recordings

To run the pipeline on days or weeks of PAM data, we've developed a parallel processing pipeline with preloading in `run_pipeline_parallel.py`, which must be called directly from the terminal with the virtual env activated.

You can call it like this from the root directory of the repository, specifying the input directory and output directory. For the input directory, it will look for `.wav` files and also check in the output folder if some files have already been processed and filter them out.

Depending on your machine, you can change `num_processes`, which defines how many copies of the pipeline will work simultaneously on different files. We have found that anything from 4 to 8 works quite well, with the main bottleneck being loading the large WAV files from a single HDD drive.

To first test out the pipeline, you can also add `--max_files 16` at the end to make sure it works:

```bash
python ./pipeline/run_pipeline_parallel.py --input_dir E:/2022/ --output_dir ./pipeline/outputs/BSM_2022 --num_processes 4
```

Thanks to the speed and parallelization, it is possible to process months of continuous PAM data in a few days.

### Pipeline Outputs

When using the parallel pipeline, the results are automatically saved in the selected `output_dir`. For each `.wav` file processed, two corresponding files will be generated:

1. **CSV file**: Contains raw outputs with a timestamp every second, including the probability of each call type and other noise and SNR values for each one-second window. (Note: These parameters are still in development and are not part of the paper. The main trustworthy outputs are the call classifications.)

2. **JSON file**: Contains a basic summary of the processed file, including processing time, number of each call type detected, etc.

When the pipeline has processed all files in `input_dir`, it will generate two final files:

- **`_merged_results.csv`**: Contains all the CSV files merged together
- **`_continuous_segments.csv`**: Contains the start and end times of continuous segments in the PAM data, and shows any gaps between files (which could be due to hydrophone maintenance or other reasons). This file provides a quick overview of data continuity and can be useful for downstream analysis.

