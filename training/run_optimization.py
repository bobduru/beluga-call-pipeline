from dotenv import load_dotenv
from pathlib import Path
import sys
import os

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

import torch
from models.resnet import ResnetMultilabel
from models.mobilenet import MobileNetMultilabel
from models.quant_mobilenet import load_mobilenet_v3_quant

from training.cross_validation import run_cross_val, train_model


labels_df = pd.read_csv("data/Verified_Dataset/labels/labels_merged.csv")
labels_df["ClipFilenamePt"] = labels_df["clip_filename"].str.replace(".wav", ".pt", regex=False)

label_columns = ["ECHO", "HFPC", "BBPC", "Whistle"]

data_dir = "data"
processed_spects_dir = data_dir + "/Verified_Dataset/spectrograms/"

results_dir = "training/new_results/optimization_last"

# labels_df["Boat"] = labels_df["Boat"].astype("boolean")
labels_df["Site_B"] = (
    labels_df["Site"] + "_B_" + labels_df["Boat"].astype("string")
).where(labels_df["Boat"].notna(), pd.NA)



training_config_default = {
    "batch_size": 32,
    "lr_decay_factor": 0.5,
    "patience_lr": 2,
    # "n_epochs": 1, #100
    # "min_epochs": 0, #15
    "n_epochs": 100, #100
    "min_epochs": 10, #15
    "patience_early_stopping": 5,
    "metric_mode": "max",
    "val_metric": "f1",
}
use_augmentation = False

run_cross_val(
    labels_df, 
    label_columns, 
    ResnetMultilabel,  
    processed_spects_dir,
    run_name="resnet",
    results_dir=results_dir,
    model_kwargs={
        "pretrained":True,
    }, 
    training_config=training_config_default,
    save_models=True,
    use_quantization=False,
    test_cols_metrics=["Site", "Boat"],
    fold_exclusive_col="labeled_snippet_filename",
    use_augmentation=use_augmentation,
)


n_layers_to_test = [8, 10, 12, 6, 4]  
# n_layers_to_test = [2, 4, 6, 8, 10, 12,]  
# quantization_options = [False,True]
quantization_options = [False, True]

for n_layers in n_layers_to_test:
    for use_quantization in quantization_options:
        # Create run name based on parameters
        quant_suffix = "_qat" if use_quantization else ""
        run_name = f"mobile_net{quant_suffix}_{n_layers}_layers"
        
        print(f"\n{'='*80}")
        print(f"Running experiment: {run_name}")
        print(f"n_layers: {n_layers}, quantization: {use_quantization}")
        print(f"{'='*80}")

        model_class = load_mobilenet_v3_quant if use_quantization else MobileNetMultilabel

        model_kwargs = {
            "pretrained": True,
            "n_layers": n_layers
        }

        if use_quantization:
            model_kwargs["qat"] = True
        
        
        run_cross_val(
            labels_df, 
            label_columns, 
            model_class,  
            processed_spects_dir,
            run_name=run_name,
            model_kwargs=model_kwargs, 
            n_splits=5,
            training_config=training_config_default,
            save_models=True,
            use_quantization=use_quantization,
            results_dir=results_dir,
            test_cols_metrics=["Site", "Boat"],
            stratify_cols=["Site", "Boat"],
            fold_exclusive_col="labeled_snippet_filename",
            use_augmentation=use_augmentation,
        )
        print(f"✅ Successfully completed: {run_name}")
            
        

print(f"\n{'='*80}")
print("All experiments completed!")
print(f"{'='*80}")