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
from torchvision.models.quantization.mobilenetv3 import QuantizableMobileNetV3

from models.dataset import MultiLabelDataset

from models.dataset import setup_multilabel_dataloaders
from models.trainer import MultilabelTrainer
from models.utils import aggregate_folds_testing_metrics,save_model, size_of_model

from models.utils import get_model_size, get_best_device

training_config_default = {
    "batch_size": 32,
    "lr_decay_factor": 0.5,
    "patience_lr": 2,
    "n_epochs": 100,
    "min_epochs": 10,
    "patience_early_stopping": 5,
    "metric_mode": "max",
    "val_metric": "f1",
}

def train_model(
        labels_df,
        label_columns,
        model,
        train_data,
        val_data,
        test_data,
        run_name="",
        training_config=training_config_default,
        processed_spects_dir=None,
        results_dir="./sites_results",
        fold_idx=None,
        use_quantization=False,
        save_models=False,
        base_dir=None,
        # compute_sites_metrics=True,
        test_cols_metrics=["Site"],
        use_augmentation=False,
):

    """
    Train a multilabel classification model, function can be used for a single training run or in cross validation.
    
    
    Parameters
    ----------
    labels_df : pandas.DataFrame
        Complete dataset containing all samples with their labels and metadata.
        Must include columns: 'Site', 'ClipFilenamePt', and all label columns.
        
    label_columns : list of str
        List of column names in labels_df that represent the target labels.
        These will be used for multilabel classification.
        
    model : torch.nn.Module
        The neural network model to train. Should be compatible with multilabel
        classification and the specified label_columns.
        
    train_data : pandas.DataFrame
        Training data subset from labels_df. Contains samples for model training.
        
    val_data : pandas.DataFrame
        Validation data subset from labels_df. Used for early stopping and
        hyperparameter tuning during training.
        
    test_data : pandas.DataFrame
        Test data subset from labels_df. Used for final model evaluation.
        
    run_name : str, default=""
        Name for this training run. Used to create organized output directories.
        If empty, results will be saved in the base results directory.
        
    training_config : dict, default=training_config_default
        Configuration dictionary containing training hyperparameters:
        - batch_size (int): Batch size for training (default: 32)
        - lr_decay_factor (float): Learning rate decay factor (default: 0.5)
        - patience_lr (int): Patience for learning rate reduction (default: 2)
        - n_epochs (int): Maximum number of training epochs (default: 100)
        - min_epochs (int): Minimum number of epochs before early stopping (default: 10)
        - patience_early_stopping (int): Patience for early stopping (default: 5)
        - metric_mode (str): Mode for metric optimization - "max" or "min" (default: "max")
        - val_metric (str): Validation metric to monitor (default: "f1")
        
    processed_spects_dir : str, optional
        Directory path containing preprocessed spectrogram files (.pt format).


    results_dir : str, default="./sites_results"
        Base directory for saving training results and model outputs.
        
    fold_idx : int, optional
        Fold index for cross-validation. If provided, results will be saved in
        a 'fold_{fold_idx}' subdirectory. Used for organizing cross-validation results.
        
    use_quantization : bool, default=False
        Whether to apply post-training quantization to the model for compression.
        Creates a quantized version and evaluates it on all test sites.
        If true the model has to be from load_mobilenet_v3_quant
        
    save_models : bool, default=False
        Whether to save the trained model weights to disk.
        Model will be saved as 'model.pt' in the run directory.
        
    base_dir : str, optional
        Override the base directory for saving results. If None, will be
        constructed as '{results_dir}/{run_name}'.

    compute_sites_metrics : bool, default=True
        If set to True, when the model is trained, it will be evaluated per site in the test set and the results will be saved in the training_details.json file.
    """

    # Check that necessary columns exist in all dataframes
    required_columns = label_columns + ['Site']
    missing_columns = [col for col in required_columns if col not in labels_df.columns]
    if missing_columns:
        raise ValueError(f"Labels DataFrame missing required columns: {missing_columns}")

    # Verify train/val/test have same columns as labels_df
    for name, df in [('train', train_data), ('validation', val_data), ('test', test_data)]:
        if not all(col in df.columns for col in labels_df.columns):
            raise ValueError(f"{name} data missing columns that are present in labels DataFrame")

    # Verify model type if quantization is enabled
    if use_quantization and not isinstance(model, QuantizableMobileNetV3):
        raise TypeError(
            "Model must be generated with load_mobilenet_v3_quant() when use_quantization=True"
        )


    # Creating run directory
    if base_dir is None:
        base_dir = f"{results_dir}/{run_name}/"

    if fold_idx is not None:
        run_dir = f"{base_dir}fold_{fold_idx}/"
    else:
        run_dir = base_dir
    os.makedirs(run_dir, exist_ok=True)


    # Setting up everything for training
    device = get_best_device()
    cpu = torch.device("cpu")
    weights = [1,1,1,1]
    loss_fn_gpu = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=torch.tensor(weights).to(device))
    metrics_gpu = get_metrics(len(label_columns), device)

    loss_fn_cpu = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=torch.tensor(weights).to(cpu))
    metrics_cpu = get_metrics(len(label_columns), cpu)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=training_config["metric_mode"],
        patience=training_config["patience_lr"],
        factor=training_config["lr_decay_factor"],
        threshold=1e-3,
        threshold_mode="abs",
    )

    trainer = MultilabelTrainer(model, device=device, labels_mapping=label_columns)
    train_loader, val_loader, test_loader = setup_multilabel_dataloaders(train_data, val_data, test_data, processed_spects_dir, label_columns, use_augmentation=use_augmentation)

    # Training the model
    trained_model, training_details, test_predictions_log = trainer.fit(
        train_loader,
        val_loader,
        test_loader,
        loss_fn_gpu,
        optimizer,
        lr_scheduler,
        n_epochs=training_config["n_epochs"],
        min_epochs=training_config["min_epochs"],
        labels_mapping=label_columns,
        patience_early_stopping=training_config["patience_early_stopping"],
        metrics=metrics_gpu,
        val_metric=training_config["val_metric"],
        val_metric_goal=training_config["metric_mode"],
        keep_test_logs=True,
        qat=use_quantization
    )

    # Saving the model
    if save_models:
        save_model(trained_model,f"{run_dir}/model.pt" )

    # Saving the test predictions
    test_prediction_df = pd.DataFrame(test_predictions_log)
    test_prediction_df = create_predictions_dataframe(test_predictions_log, label_columns)
    test_prediction_df["Fold"] = fold_idx
    test_prediction_df.to_csv(run_dir + "test_predictions.csv", index=False)

    # Adding extra details to the training details
    size_mb, size_kb = size_of_model(trained_model)
    training_details["model_size_mb"] = size_mb
    training_details["model_size_kb"] = size_kb
    training_details["label_mapping"] = label_columns

    training_details["other_tests"] = {}

    quant_model=None
    if use_quantization:
        # Quantizing the model (Pytorch quantization only works on cpu for now)
        model_cpu = copy.deepcopy(trained_model).to(cpu)
        model_cpu.eval()
        quant_model = torch.ao.quantization.convert(model_cpu, inplace=False)

        trainer_cpu = MultilabelTrainer(quant_model, device=torch.device("cpu"), labels_mapping=label_columns)
        quant_test_results, _ = trainer_cpu.test_epoch( 0, test_loader, loss_fn_cpu, metrics_cpu, phase=f"Test quantization fold {fold_idx}", keep_logs=True)
        
        size_mb, size_kb = size_of_model(quant_model)
        training_details["quantized_model_size_mb"] = size_mb
        training_details["quantized_model_size_kb"] = size_kb

        training_details["other_tests"]["quantized"] = quant_test_results

    # if compute_sites_metrics:   
    for test_col in test_cols_metrics:
        #Here we are evaluating the model on each site in the test set
        labels_df[test_col] = labels_df[test_col].astype("string")

        test_vals = list(set(labels_df[test_col].unique()))
        if "test_fold_idx" in labels_df.columns:
            test_df = labels_df[labels_df["test_fold_idx"] == fold_idx]
        else:
            test_df = labels_df

        for test_val in test_vals:
            test_val_df = test_df[test_df[test_col] == test_val]
            if len(test_val_df) == 0:
                print(f"No data for {test_col} {test_val}, skipping.")
                continue

            test_val_loader = DataLoader(
                MultiLabelDataset(
                    test_val_df,
                    processed_spects_dir=processed_spects_dir,
                    label_columns=label_columns,
                    train=True,
                    spect_filename_column='ClipFilenamePt'
                ),
                batch_size=training_config["batch_size"],
                shuffle=False
            )

            print(f"Created test dataloader for {test_col} {test_val} with {len(test_val_df)} samples")

            test_results, test_predictions_log = trainer.test_epoch( 0, test_val_loader, loss_fn_gpu, metrics_gpu, phase="Test on " + test_col + " " + test_val, keep_logs=True)

            test_prediction_df = create_predictions_dataframe(test_predictions_log, label_columns)
            test_prediction_df.to_csv(run_dir + f"{test_col}_{test_val}_test_predictions.csv", index=False)

            test_name = test_col + "_" + str(test_val)
            training_details["other_tests"][test_name] = test_results
            
            if use_quantization:  
                trainer_cpu = MultilabelTrainer(quant_model, device=cpu, labels_mapping=label_columns)
                quant_test_results, _ = trainer_cpu.test_epoch( 0, test_val_loader, loss_fn_cpu, metrics_cpu, phase="(Quantized) Test on " + test_col + " " + test_val, keep_logs=True)
                training_details["other_tests"][f"{test_name}_quantized"] = quant_test_results

    save_json(training_details, run_dir + "training_details.json")

    return base_dir, test_prediction_df, training_details


# Create test fold indices where each site has the same number of test samples


def run_cross_val(
    labels_df, 
    label_columns, 
    model_class, 
    processed_spects_dir,
    run_name,
    model_kwargs={}, 
    n_splits=5,
    training_config=training_config_default,
    save_models=False,
    use_quantization=False, 
    results_dir="./final_results",
    test_cols_metrics=["Site"],
    fold_exclusive_col="labeled_snippet_filename",
    use_augmentation=False,
    stratify_cols=None,
):
    """""
    
    This function performs nested k-fold cross-validation where each fold maintains
    the same distribution of sites in the test set. 
    
    Parameters
    ----------
    labels_df : pandas.DataFrame
        Complete dataset containing all samples with their labels and metadata.
        Must include columns: 'Site', 'ClipFilenamePt', and all label columns.
        The 'Site' column is used for stratified splitting to ensure balanced
        representation across folds.
        
    label_columns : list of str
        List of column names in labels_df that represent the target labels for
        multilabel classification. These will be used to determine the number
        of output classes for the model.
        
    model_class : class
        The neural network model class to instantiate for each fold. Must be
        callable with `model_class(num_classes=len(label_columns), **model_kwargs)`.
        Examples: MobileNetMultilabel, ResnetMultilabel
        
    processed_spects_dir : str
        Directory path containing preprocessed spectrogram files (.pt format).
        Used by the MultiLabelDataset to load input features.
        
    run_name : str
        Unique name for this cross-validation experiment. Used to create organized
        output directories and distinguish between different experimental runs.
        
    model_kwargs : dict, default={}
        Additional keyword arguments to pass to the model constructor.
        Common examples: {'pretrained': True, 'n_layers': 8, 'dense_layer_size': 256}
        
    n_splits : int, default=5
        Number of folds for cross-validation. Each fold will have approximately
        1/n_splits of the data as test set, with the rest used for training/validation.
        
    training_config : dict, default=training_config_default
        Configuration dictionary containing training hyperparameters:
        - batch_size (int): Batch size for training (default: 32)
        - lr_decay_factor (float): Learning rate decay factor (default: 0.5)
        - patience_lr (int): Patience for learning rate reduction (default: 2)
        - n_epochs (int): Maximum number of training epochs (default: 100)
        - min_epochs (int): Minimum number of epochs before early stopping (default: 10)
        - patience_early_stopping (int): Patience for early stopping (default: 5)
        - metric_mode (str): Mode for metric optimization - "max" or "min" (default: "max")
        - val_metric (str): Validation metric to monitor (default: "f1")
        
    save_models : bool, default=False
        Whether to save the trained model weights for each fold.
        Models will be saved as 'model.pt' in each fold's directory.
        
    use_quantization : bool, default=False
        Whether to apply post-training quantization to models for compression.
        Creates quantized versions and evaluates them on all test sites.
        
    results_dir : str, default="./final_results"
        Base directory for saving all cross-validation results.
        Results will be organized as: {results_dir}/{run_name}/fold_{i}/
        
    compute_sites_metrics : bool, default=True
        Whether to compute site-specific metrics for generalization analysis.
        If True, evaluates the model on each individual site in the test set.
        
"""

    base_dir = results_dir + "/" + run_name + "/"
    os.makedirs(base_dir, exist_ok=True)

    test_predictions_dfs = []


    # Create test fold indices where each site has the same number of test samples
    labels_df = create_test_fold_indices(labels_df, n_splits=n_splits, group_col=fold_exclusive_col, stratify_cols=stratify_cols)
    print(labels_df.groupby(['test_fold_idx', 'Site']).size())
    print(labels_df.groupby(['test_fold_idx', 'Boat']).size())

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
            training_config=training_config,
            use_quantization=use_quantization,
            save_models=save_models,
            test_cols_metrics=test_cols_metrics,
            use_augmentation=use_augmentation,
        )

        test_predictions_dfs.append(test_prediction_df)


    test_predictions_df = pd.concat(test_predictions_dfs, ignore_index=True)
    test_predictions_df.to_csv(base_dir + "test_predictions.csv", index=False)

    aggregate_folds_testing_metrics(base_dir)

    return test_predictions_df


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

# def create_test_fold_indices(labels_df, n_splits=5):
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#     labels_df['test_fold_idx'] = -1 
#     for fold_idx, (_, test_idx) in enumerate(skf.split(labels_df, labels_df['Site'])):
#         labels_df.loc[test_idx, 'test_fold_idx'] = fold_idx
#     return labels_df


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def create_test_fold_indices(
    labels_df: pd.DataFrame,
    *,
    n_splits: int = 5,
    stratify_col: str = "Site",
    stratify_cols: list[str] | None = None,  # NEW: multiple stratification columns
    group_col: str | None = None,
    random_state: int = 42,
    out_col: str = "test_fold_idx",
) -> pd.DataFrame:
    """
    If group_col is provided:
      - non-NaN groups are kept together (avoid leakage) using StratifiedGroupKFold if available
      - NaN rows are assigned individually (stratified-ish) while balancing fold sizes
    If group_col is None:
      - simple row-level StratifiedKFold (your original behavior)
    
    NEW: stratify_cols allows stratifying on multiple columns simultaneously
    """
    df = labels_df.copy()
    df[out_col] = -1

    # Handle multiple stratification columns
    if stratify_cols is not None:
        # Create combined stratification column (element-wise)
        for col in stratify_cols:
            if col not in df.columns:
                raise KeyError(f"stratify_col='{col}' not found in df.columns")
        
        # Combine columns element-wise by building string progressively
        # Convert to string first (handles all dtypes), then replace 'nan' string
        df['_combined_strat'] = df[stratify_cols[0]].fillna('__NA__').astype(str)
        for col in stratify_cols[1:]:
            df['_combined_strat'] = df['_combined_strat'] + '_' + df[col].fillna('__NA__').astype(str)
        
        effective_stratify_col = '_combined_strat'
    else:
        # Single stratification column (backward compatible)
        if stratify_col not in df.columns:
            raise KeyError(f"stratify_col='{stratify_col}' not found in df.columns")
        effective_stratify_col = stratify_col
    
    if group_col is not None and group_col not in df.columns:
        raise KeyError(f"group_col='{group_col}' not found in df.columns")

    rng = np.random.RandomState(random_state)

    # ----------------------------
    # Case A: no grouping -> original behavior
    # ----------------------------
    if group_col is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for fold_idx, (_, test_idx) in enumerate(skf.split(df, df[effective_stratify_col])):
            df.iloc[test_idx, df.columns.get_loc(out_col)] = fold_idx
        
        # Clean up temporary column
        if stratify_cols is not None:
            df = df.drop(columns=['_combined_strat'])
        return df

    # ----------------------------
    # Case B: grouping enabled
    # ----------------------------
    mask_grouped = df[group_col].notna()
    df_g = df[mask_grouped]
    df_n = df[~mask_grouped]

    # 1) Grouped (non-NaN)
    if len(df_g) > 0:
        try:
            from sklearn.model_selection import StratifiedGroupKFold

            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            for fold_idx, (_, test_idx_local) in enumerate(
                sgkf.split(df_g, y=df_g[effective_stratify_col].values, groups=df_g[group_col].values)
            ):
                df.loc[df_g.iloc[test_idx_local].index, out_col] = fold_idx

        except Exception as e:
            # Greedy fallback (balances class + total size)
            print(f"StratifiedGroupKFold failed: {e}, using greedy fallback")
            y = df_g[effective_stratify_col].astype(str).values
            groups = df_g[group_col].astype(str).values
            classes = np.unique(y)

            group_to_idx = {}
            group_to_counts = {}
            for idx, (g, c) in zip(df_g.index, zip(groups, y)):
                group_to_idx.setdefault(g, []).append(idx)
                if g not in group_to_counts:
                    group_to_counts[g] = {cl: 0 for cl in classes}
                group_to_counts[g][c] += 1

            total_counts = {cl: int((y == cl).sum()) for cl in classes}
            target_per_fold = {cl: total_counts[cl] / n_splits for cl in classes}
            target_size = len(df_g) / n_splits

            fold_counts = [{cl: 0 for cl in classes} for _ in range(n_splits)]
            fold_sizes = [0 for _ in range(n_splits)]

            group_list = list(group_to_idx.keys())
            group_list.sort(key=lambda g: len(group_to_idx[g]), reverse=True)

            # shuffle ties deterministically
            i = 0
            while i < len(group_list):
                j = i
                sz = len(group_to_idx[group_list[i]])
                while j < len(group_list) and len(group_to_idx[group_list[j]]) == sz:
                    j += 1
                if j - i > 1:
                    chunk = group_list[i:j]
                    rng.shuffle(chunk)
                    group_list[i:j] = chunk
                i = j

            def fold_cost(k, g_counts, g_size):
                cost = 0.0
                for cl in classes:
                    new_val = fold_counts[k][cl] + g_counts[cl]
                    cost += (new_val - target_per_fold[cl]) ** 2
                new_size = fold_sizes[k] + g_size
                cost += 0.25 * (new_size - target_size) ** 2
                return cost

            for g in group_list:
                g_idx = group_to_idx[g]
                g_size = len(g_idx)
                g_counts = group_to_counts[g]
                best_fold = min(range(n_splits), key=lambda k: fold_cost(k, g_counts, g_size))

                df.loc[g_idx, out_col] = best_fold
                fold_sizes[best_fold] += g_size
                for cl in classes:
                    fold_counts[best_fold][cl] += g_counts[cl]

    # 2) NaN rows: assign individually, balancing final fold sizes
    if len(df_n) > 0:
        current_sizes = df[out_col].value_counts().to_dict()
        fold_sizes = [int(current_sizes.get(k, 0)) for k in range(n_splits)]

        y_n = df_n[effective_stratify_col].astype(str)
        for cl in y_n.unique():
            idxs = df_n.index[y_n == cl].to_numpy()
            idxs = np.array(idxs, copy=True)
            rng.shuffle(idxs)

            for idx in idxs:
                min_size = min(fold_sizes)
                candidates = [k for k, sz in enumerate(fold_sizes) if sz == min_size]
                chosen = candidates[rng.randint(len(candidates))]
                df.loc[idx, out_col] = chosen
                fold_sizes[chosen] += 1

    if (df[out_col] == -1).any():
        missing = int((df[out_col] == -1).sum())
        print(f"Missing rows debug info:")
        print(f"Total rows: {len(df)}")
        print(f"Grouped rows (df_g): {len(df_g) if 'df_g' in locals() and len(df_g) > 0 else 0}")
        print(f"Non-grouped rows (df_n): {len(df_n) if 'df_n' in locals() and len(df_n) > 0 else 0}")
        print(f"Rows with fold assigned: {(df[out_col] != -1).sum()}")
        if group_col:
            print(f"\nUnassigned rows sample:")
            print(df[df[out_col] == -1][[group_col, effective_stratify_col]].head(10))
        else:
            print(f"\nUnassigned rows sample:")
            print(df[df[out_col] == -1][[effective_stratify_col]].head(10))
        raise RuntimeError(f"{missing} rows were not assigned a fold (still -1).")

    # Clean up temporary column
    if stratify_cols is not None:
        df = df.drop(columns=['_combined_strat'])
    
    return df