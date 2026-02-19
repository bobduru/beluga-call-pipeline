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

results_dir = "training/results/"

# labels_df["Boat"] = labels_df["Boat"].astype("boolean")
labels_df["Site_B"] = (
    labels_df["Site"] + "_B_" + labels_df["Boat"].astype("string")
).where(labels_df["Boat"].notna(), pd.NA)


training_config_default = {
    "batch_size": 32,
    "lr_decay_factor": 0.5,
    "patience_lr": 2,
    "n_epochs": 1, #100
    "min_epochs": 0, #15
    # "n_epochs": 100, #100
    # "min_epochs": 10, #15
    "patience_early_stopping": 5,
    "metric_mode": "max",
    "val_metric": "f1",
}



from training.cross_validation import create_test_fold_indices
from sklearn.model_selection import KFold, train_test_split
from models.utils import aggregate_folds_testing_metrics


# labels_df = create_test_fold_indices(labels_df, 5, group_col="labeled_snippet_filename")

########################################################
# LEAVE ONE SITE OUT
########################################################


# all_sites = ["BSM", "RDL", "CAC", "KAM" ]
all_sites = ["BSM","CAC", "KAM" ]

# quantization_options = [False, True]
quantization_options = [False]

max_sample_size = 500
sample_sizes = [0, 100, 200, 300, 400, max_sample_size]

for use_quantization in quantization_options:

    for out_site in all_sites:
        # for fold_idx in range(5):
        print(f"Training model for site {out_site}")
        model_class = load_mobilenet_v3_quant if use_quantization else MobileNetMultilabel

        model = model_class(
            pretrained=True,
            n_layers=8,
            num_classes=len(label_columns)
        )

        train_sites = [site for site in all_sites if site != out_site]

        test_site_df = labels_df[labels_df["Site"] == out_site]

        if len(test_site_df) > max_sample_size:
            test_site_train_sample = test_site_df.sample(n=max_sample_size, random_state=42)
            test_data = test_site_df.drop(test_site_train_sample.index)
        else:
            print(f"ERROR: Test site {out_site} has less than {max_sample_size} samples")
            continue


        train_sites_df = labels_df[labels_df["Site"].isin(train_sites)]

        
        for sample_size in sample_sizes:
            cur_sample = test_site_train_sample.sample(n=sample_size, random_state=42)

            train_df = pd.concat([train_sites_df, cur_sample])

            train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['Site'])
            

            run_name = f"leave_{out_site}_out_include_{sample_size}"
            if use_quantization:
                run_name = run_name + "_qat"

            print(run_name)
            print(f"Site out : {out_site}")
            print("Train df")
            print(train_data["Site"].value_counts())
            print("\nVal df")
            print(val_data["Site"].value_counts())

            results_dir = f"./training/results/sites_generalization_include_sample/"

            # break
            run_dir, _, _ = train_model(
                labels_df,
                label_columns,
                model,
                train_data,
                val_data,
                test_data,
                processed_spects_dir=processed_spects_dir,
                fold_idx=None,
                run_name=run_name,
                results_dir=results_dir,
                training_config=training_config_default,
                use_quantization=use_quantization,
                use_augmentation=False,
                test_cols_metrics=[],
            )

            cur_sample.to_csv(f"{results_dir}/{run_name}/test_site_train_sample.csv", index=False)
        
        # aggregate_folds_testing_metrics(run_dir)

    # break