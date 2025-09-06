# Automated Classification of Beluga Whale Diphonic Vocalizations Using Multi-Output ResNet

### This project explores the application of Passive Acoustic Monitoring (PAM) combined with deep learning techniques to enhance the real-time classification of beluga whale vocalizations in the St. Lawrence Estuary. Using a Multi-Output ResNet model, the goal is to classify both high-frequency calls and low-frequency whistles.

### We've also developed an automatic pipeline that first detects regions of interest (ROIs), then uses the call type model to predict which type of call is present.

---

### Installation

To install the necessary dependencies for this project, follow these steps:

1. Make sure Python is properly installed on your machine.

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   ```
   ```bash
   source .venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

### Data

The data used in this project is owned by the LISSE Lab in Quebec. To access it, please contact [projetbruit2020@gmail.com](mailto:projetbruit2020@gmail.com).

---

### Setup

After installing the dependencies, create a `.env` file in the root directory and set:

```
PROJECT_ROOT=/your/root/dir/beluga-call-pipeline
```

This is then used in each `.ipynb` file to make sure imports work correctly across the project, using this snippet:

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

### Basic Tips for the Code

The folders and files should be mostly self-explanatory. Core logic is in `.py` files, while most of the main runs and tests are done in `.ipynb` notebooks.

Files ending in `_old` or `_test` can be ignored for now — I left them in case I need to refer back to them later.

- The `training` folder is for training and running experiments with the call type model. Each experiment has a corresponding folder in `results`, with an `analysis.ipynb` to visualize the outcome.
- The `preprocessing` folder is to preprocess the raw audios into spectrograms to be used to train the call type model.
- The `spectrogram` folder contains the single class instance `SPECT_GENERATOR`, that is used through out the code to generate the spectrograms, there is also the `HYDROPHONE_SENSITIVITY` to load the sensitivities of known hydrophones.
- The `pipeline` folder contains the pipeline logic.
- The `ROI` folder contains the ROI logic.
- The `abundance` folder still needs cleaning.

- The `AVES` folder is outdated — tests were run but results weren’t promising. Kept for reference.
- The `custom_resnet` folder includes a ResNet implementation used by Xavier Secheresse and Tristan Cotillard. I left it here for reference, but I mainly used the models from PyTorch.

---

### Running the Pipeline

There are two main functions for running the pipeline:
- `run_pipeline_single_audio`: to run the pipeline on a single audio file. I recommend trying this first with `debug=True` to see how it operates.

- `run_multi_file_pipeline`: to run the pipeline on a folder containing multiple files.

Check out `run.ipynb` for an example of how to use them. I recommend creating your own `run_<name>.ipynb` files to organize your tests.

**Note:** If you're working with audio files recorded by a hydrophone model that isn't listed in `./spectrogram/hydrophones_sensitivity.json`, you need to manually add its sensitivity value.

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

This value is necessary for converting the spectrograms into dB re 1 µPa. If this is not done things will still work as a default sensitivity of -170 will be set, but the resulting dB values of the pipeline won't be exact.



