
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import F1Score, Accuracy, ConfusionMatrix, Precision, Recall, ExactMatch, AUROC
import copy

import json

from sklearn.model_selection import StratifiedKFold

from models.resnet import ResnetMultilabel
from models.mobilenet import MobileNetMultilabel
from models.dataset import MultiLabelDataset

from training.utils import get_model_size, get_best_device


def get_metrics(num_classes, device):

    metrics = {
        "accuracy": Accuracy(
            num_labels=num_classes, average="macro", task="multilabel"
        ).to(device),
        "multilabel_accuracy": Accuracy(
            num_labels=num_classes, average=None, task="multilabel"
        ).to(device),
        "f1": F1Score(num_labels=num_classes, average="macro", task="multilabel").to(
            device
        ),
        "multilabel_f1": F1Score(
            num_labels=num_classes, average=None, task="multilabel"
        ).to(device),
        "precision": Precision(
            num_labels=num_classes, average="macro", task="multilabel"
        ).to(device),
        "multilabel_precision": Precision(
            num_labels=num_classes, average=None, task="multilabel"
        ).to(device),
        "recall": Recall(num_labels=num_classes, average="macro", task="multilabel").to(
            device
        ),
        "multilabel_recall": Recall(
            num_labels=num_classes, average=None, task="multilabel"
        ).to(device),
        "AUC": AUROC(task="multilabel", num_labels=num_classes).to(device),
        "multilabel_AUC": AUROC(
            task="multilabel", num_labels=num_classes, average=None
        ).to(device),
        "multilabel_confusion_matrix": ConfusionMatrix(
            task="multilabel", num_labels=num_classes
        ).to(device),
        "exact_match": ExactMatch(
            task="multilabel", num_labels=num_classes, multidim_average="global"
        ).to(device),
    }

    return metrics

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)


def create_predictions_dataframe(test_predictions_log, label_columns):
    rows = []
    for d in test_predictions_log:
        row = {"Filename": d["Filename"], "Site": d["Site"]}
        for i, label in enumerate(label_columns):
            row[f"{label}_true"] = d["true"][i]
            row[f"{label}_pred"] = d["pred"][i] 
            row[f"{label}_probs"] = d["probs"][i]
        rows.append(row)

    # Create DataFrame
    all_calls_df = pd.DataFrame(rows)
    
    print(all_calls_df.head())
    return all_calls_df


from models.dataset import setup_multilabel_dataloaders
from models.trainer import MultilabelTrainer
from models.utils import aggregate_folds_testing_metrics,save_model, size_of_model
from models.quant_mobilenet import load_mobilenet_v3_quant, quantize_model_post_training



def train_model(
        labels_df,
        label_columns,
        model,
        train_data,
        val_data,
        test_data,
        use_weights=False,
        run_name="",
        metric_mode="max",
        val_metric="f1",
        patience_lr=2,
        lr_decay_factor=0.5,
        n_epochs=100,
        min_epochs=10,
        processed_spects_dir=None,
        batch_size=32,
        results_dir="./sites_results",
        fold_idx=None,
        patience_early_stopping=5,
        use_quantization=False,
        save_models=False,
        base_dir=None
):
    device = get_best_device()
    cpu = torch.device("cpu")

    weights = [1,2,2,1] if use_weights else [1,1,1,1]
    loss_fn_gpu = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=torch.tensor(weights).to(device))
    metrics_gpu = get_metrics(len(label_columns), device)

    loss_fn_cpu = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=torch.tensor(weights).to(cpu))
    metrics_cpu = get_metrics(len(label_columns), cpu)


    sites = list(set(labels_df["Site"].unique()))

    if base_dir is None:
        base_dir = f"{results_dir}/{run_name}"

    if fold_idx is not None:
        run_dir = f"{base_dir}/fold_{fold_idx}/"
    else:
        run_dir = base_dir
    
    os.makedirs(run_dir, exist_ok=True)

    test_df = labels_df[labels_df["test_fold_idx"] == fold_idx]
    sites_test_loaders = {}

    for site in sites:
        site_test_df = test_df[test_df["Site"] == site]

        if len(site_test_df) == 0:
            print(f"No data for site {site}, skipping.")
            continue

        site_test_loader = DataLoader(
            MultiLabelDataset(
                site_test_df,
                processed_spects_dir=processed_spects_dir,
                label_columns=label_columns,
                train=True,
                spect_filename_column='ClipFilenamePt'
            ),
            batch_size=batch_size,
            shuffle=False
        )

        sites_test_loaders[site] = site_test_loader


    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=metric_mode,
        patience=patience_lr,
        factor=lr_decay_factor,
        threshold=1e-3,
        threshold_mode="abs",
    )

    trainer = MultilabelTrainer(model, device=device, labels_mapping=label_columns)

    train_loader, val_loader, test_loader = setup_multilabel_dataloaders(train_data, val_data, test_data, processed_spects_dir, label_columns)

    trained_model, training_details, test_predictions_log = trainer.fit(
        train_loader,
        val_loader,
        test_loader,
        loss_fn_gpu,
        optimizer,
        lr_scheduler,
        n_epochs,
        min_epochs=min_epochs,
        labels_mapping=label_columns,
        patience_early_stopping=patience_early_stopping,
        metrics=metrics_gpu,
        val_metric=val_metric,
        val_metric_goal=metric_mode,
        keep_test_logs=True,
        qat=use_quantization
    )

    if save_models:
        save_model(trained_model,f"{run_dir}/model.pt" )

    test_prediction_df = pd.DataFrame(test_predictions_log)
    test_prediction_df = create_predictions_dataframe(test_predictions_log, label_columns)
    test_prediction_df["Fold"] = fold_idx
    test_prediction_df.to_csv(run_dir + "test_predictions.csv", index=False)


    size_mb, size_kb = size_of_model(trained_model)
    training_details["model_size_mb"] = size_mb
    training_details["model_size_kb"] = size_kb
    training_details["label_mapping"] = label_columns

    training_details["other_tests"] = {}
    
    quant_model=None

    if use_quantization:
        model_cpu = copy.deepcopy(trained_model).to(cpu)
        model_cpu.eval()
        quant_model = torch.ao.quantization.convert(model_cpu, inplace=False)

     # Loop through test_sites and create a dataloader for each
    for site in sites:
        site_test_loader = sites_test_loaders[site]

        print(f"Created test dataloader for site: {site} with {len(site_test_df)} samples")
        # --- Add your test code here, e.g. run model on site_test_loader ---
        test_results, test_predictions_log = trainer.test_epoch( 0, site_test_loader, loss_fn_gpu, metrics_gpu, phase="Test on site " + site, keep_logs=True)
        
        test_prediction_df = create_predictions_dataframe(test_predictions_log, label_columns)
        test_prediction_df.to_csv(run_dir + f"{site}_test_predictions.csv", index=False)

        training_details["other_tests"][site] = test_results

        if use_quantization:  
            trainer_cpu = MultilabelTrainer(quant_model, device=cpu, labels_mapping=label_columns)
            quant_test_results, _ = trainer_cpu.test_epoch( 0, site_test_loader, loss_fn_cpu, metrics_cpu, phase="(Quantized) Test on site " + site, keep_logs=True)
            training_details["other_tests"][f"{site}_quantized"] = quant_test_results
        
        
    save_json(training_details, run_dir + "training_details.json")

    return base_dir, test_prediction_df, training_details


 # Create test fold indices where each site has the same number of test samples
def create_test_fold_indices(labels_df, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    labels_df['test_fold_idx'] = -1 
    for fold_idx, (_, test_idx) in enumerate(skf.split(labels_df, labels_df['Site'])):
        labels_df.loc[test_idx, 'test_fold_idx'] = fold_idx
    return labels_df


def run_cross_val(
    labels_df, 
    label_columns, 
    model_class, 
    processed_spects_dir,
    run_name,
    model_kwargs={}, 
    n_splits=5,
    batch_size=32,
    lr_decay_factor=0.5,
    patience_lr=2,
    n_epochs=50,
    metric_mode="max",
    val_metric="f1",
    save_models=False,
    use_weights=False,
    use_quantization=False, 
    results_dir="./final_results"
):

    base_dir = results_dir + "/" + run_name + "/"
    os.makedirs(base_dir, exist_ok=True)

    test_predictions_dfs = []


    # Create test fold indices where each site has the same number of test samples
    labels_df = create_test_fold_indices(labels_df, n_splits)


    for fold_idx in range(n_splits):
        model = model_class(num_classes=len(label_columns), **model_kwargs)

        print(f"Training model for fold {fold_idx + 1} of {n_splits}")

        train_data = labels_df[labels_df["test_fold_idx"] != fold_idx]
        test_data = labels_df[labels_df["test_fold_idx"] == fold_idx]

        train_data, val_data = train_test_split(train_data, test_size=0.2)

        print(f"\nFold {fold_idx + 1} dataset sizes:")
        print(f"Train: {len(train_data)} samples")
        print(f"Val: {len(val_data)} samples") 
        print(f"Test: {len(test_data)} samples\n")


        base_dir, test_prediction_df,training_details = train_model(
            labels_df,
            label_columns,
            model,
            train_data,
            val_data,
            test_data,
            fold_idx=fold_idx,
            processed_spects_dir=processed_spects_dir,
            base_dir=base_dir,
            batch_size=batch_size,
            lr_decay_factor=lr_decay_factor,
            patience_lr=patience_lr,
            n_epochs=n_epochs,
            metric_mode=metric_mode, 
            val_metric=val_metric,
            use_weights=use_weights,
            use_quantization=use_quantization,
            save_models=save_models
        )

        test_predictions_dfs.append(test_prediction_df)


    test_predictions_df = pd.concat(test_predictions_dfs, ignore_index=True)
    test_predictions_df.to_csv(base_dir + "test_predictions.csv", index=False)

    aggregate_folds_testing_metrics(base_dir)

    return test_predictions_df
