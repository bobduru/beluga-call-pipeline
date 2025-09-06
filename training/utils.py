import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import F1Score, Accuracy, MulticlassConfusionMatrix, Precision, Recall


import json




def filter_df_by_existing_files(directory, df, filename_column='SnippetFilename'):
    """
    Filters the DataFrame by removing rows where the corresponding file in the directory does not exist.
    
    Parameters:
    - directory (str): The directory where the .pt files are stored.
    - df (pd.DataFrame): The DataFrame containing the filenames.
    - filename_column (str): The column in the DataFrame that contains the filenames (default is 'SnippetFilename').
    
    Returns:
    - pd.DataFrame: A filtered DataFrame with only the rows where the corresponding .pt file exists in the directory.
    """
    # Make a copy of the DataFrame to avoid modifying the original DataFrame
    df_filtered = df.copy()
    
    # Function to check if the corresponding .pt file exists
    def file_exists(row):
        wav_filename = row[filename_column]
        pt_filename = os.path.splitext(wav_filename)[0] + '.pt'
        return os.path.exists(os.path.join(directory, pt_filename))
    
    # Filter the DataFrame
    df_filtered = df_filtered[df_filtered.apply(file_exists, axis=1)]
    
    return df_filtered

def split_data(
    df, split_fracs={"train": 0.6, "val": 0.2, "test": 0.2}, random_state=None
):
    # Ensure the split fractions sum to 1
    assert sum(split_fracs.values()) == 1.0, "Split fractions must sum to 1.0"

    # Split the data into training and temp sets
    train_data, temp_data = train_test_split(
        df, test_size=1 - split_fracs["train"], random_state=random_state
    )

    # Calculate validation fraction relative to the remaining data
    val_frac = split_fracs["val"] / (split_fracs["val"] + split_fracs["test"])

    # Split the temp data into validation and test sets
    val_data, test_data = train_test_split(
        temp_data, test_size=1 - val_frac, random_state=random_state
    )

    return (
        train_data.reset_index(drop=True, inplace=False),
        val_data.reset_index(drop=True, inplace=False),
        test_data.reset_index(drop=True, inplace=False),
    )

def preprocess_df(df, drop_others, merge_hfpcs):
    df = df[df["HF_Call"] != "W"]
    df = df[df["HF_Call"] != "w"]

    if merge_hfpcs:
        df.loc[ df["HF_Call"] == "HFPC-M", "HF_Call" ] = "HFPC"

    if drop_others:
        df = df[df["HF_Call"] != "OTHER"]
    return df

def downsample(df, label_name = "HF_Call_Code", n_samples=None):
    """
    Balances the class distribution within a DataFrame by downsampling or taking all instances
    from classes that cannot be downsampled to the desired number of samples.

    Parameters:
    - df (pandas.DataFrame): DataFrame with a 'Label' column.
    - n_samples (int, optional): Number of samples per class; defaults to the size of the smallest class.

    Returns:
    - pandas.DataFrame: A DataFrame where each class has either been downsampled or all instances are taken.
    """
    if n_samples is None:
        n_samples = df[label_name].value_counts().min()

    label_counts = df[label_name].value_counts()
    downsampled_df = pd.DataFrame()
    
    for label, count in label_counts.items():
        if count < n_samples:
            n_samples_adjusted = count  # Take all samples if not enough to downsample
        else:
            n_samples_adjusted = n_samples
        
        sampled_df = df[df[label_name] == label].sample(n=n_samples_adjusted, random_state=42)
        downsampled_df = pd.concat([downsampled_df, sampled_df], axis=0)

    return downsampled_df.reset_index(drop=True)

def get_class_weights(df, label_name = "HF_Call_Code"):

    y = df[label_name]
    label_counts = y.value_counts()
    n_max = label_counts.max()
    weights = n_max / label_counts
    weights = weights.sort_index()
    
    # Convert weights to a PyTorch tensor
    class_weights = torch.tensor(weights.values, dtype=torch.float32)
    return class_weights

def setup_dataloaders(train_data, val_data, test_data,processed_files_dir, call_type_label_column="HF_Call_Code", whistle_label_column="Whistle", batch_size=32, resize_1_5s=False):
    train_dataset = MultiHeadedDataset(
        train_data,
        processed_files_dir=processed_files_dir,
        call_type_label_column=call_type_label_column,
        whistle_label_column=whistle_label_column,
        resize_1_5s=resize_1_5s,
        train=True
    )
    val_dataset = MultiHeadedDataset(
        val_data,
        processed_files_dir=processed_files_dir,
        call_type_label_column=call_type_label_column,
        whistle_label_column=whistle_label_column,
        resize_1_5s=resize_1_5s
    )
    test_dataset = MultiHeadedDataset(
        test_data,
        processed_files_dir=processed_files_dir,
        call_type_label_column=call_type_label_column,
        whistle_label_column=whistle_label_column,
        resize_1_5s=resize_1_5s
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader




import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix, classification_report

def test_report(
    results_df,
    class_names=["ECHO", "HFPC", "CC", "Whistle"],
    filter_absent=True,
    show_cms=False,
    add_abs_column=True,
    show_report=True
):
    """
    Display multilabel confusion matrices for each class.
    
    Args:
        results_df: DataFrame with columns like 'ECHO_true', 'ECHO_pred', etc.
        class_names: List of class names.
        filter_absent: If True, only include rows where at least one class is present in the ground truth.
    """
    # Add Absence columns for true and pred
    results_df = results_df.copy()
    
    all_class_names = class_names
    # Optionally filter out rows where all true labels are 0 (fully absent)
    if filter_absent:
        mask = (results_df[[f"{c}_true" for c in class_names]].sum(axis=1) > 0)
        filtered_df = results_df[mask]
    else:
        if add_abs_column:
            results_df["Abs_true"] = (results_df[[f"{c}_true" for c in class_names]].sum(axis=1) == 0).astype(int)
            results_df["Abs_pred"] = (results_df[[f"{c}_pred" for c in class_names]].sum(axis=1) == 0).astype(int)
            all_class_names = class_names + ["Abs"]
        filtered_df = results_df

    y_true = filtered_df[[f"{c}_true" for c in all_class_names]].astype(int).values
    y_pred = filtered_df[[f"{c}_pred" for c in all_class_names]].astype(int).values

    no_preds = (y_pred.sum(axis=1) == 0).sum()
    # print(f"{no_preds} samples had no predicted labels.")

    mcm = multilabel_confusion_matrix(y_true, y_pred)

    report = classification_report(y_true, y_pred, target_names=all_class_names, output_dict=True, zero_division=0)

    if show_report: 
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=all_class_names, zero_division=0))
    
    
    if show_cms:
        for i, class_name in enumerate(class_names):
            cm = mcm[i]
            plt.figure(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=["Pred 0", "Pred 1"], 
                        yticklabels=["True 0", "True 1"])
            metrics = report[class_name]
            title = (f"Confusion Matrix for {class_name}, {"only with calls" if filter_absent else "with full absences"}\n"
                    f"precision: {metrics['precision']:.3f}\n"
                    f"recall: {metrics['recall']:.3f}\n" 
                    f"f1-score: {metrics['f1-score']:.3f}\n")
            plt.title(title)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.show()

    return report
    
def load_training_details(runs_list, base_path="./"):
    """
    Load training_details.json files from each run directory or a single run directory.

    Args:
        runs_list: List of run directory names or a single run directory name (str or list)
        base_path: Base path where run directories are located

    Returns:
        If a single run is provided (str), returns the loaded JSON data (dict or list).
        If a list is provided, returns a dictionary with run names as keys and loaded JSON data as values.
    """
    # If a single run is provided as a string, convert to list for processing
    is_single = False
    if isinstance(runs_list, str):
        runs_list = [runs_list]
        is_single = True

    training_details = {}

    for run in runs_list:
        json_path = os.path.join(base_path, run, "training_details.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as file:
                    training_details[run] = json.load(file)
            except Exception as e:
                print(f"Error loading {json_path}: {e}")
        else:
            print(f"Warning: {json_path} does not exist")

    # If only one run was provided, return its details directly
    if is_single:
        return training_details[runs_list[0]] if runs_list[0] in training_details else None
    return training_details

def create_multiclass_confusion_matrix(results_df, call_types=['ECHO', 'HFPC', 'CC', 'Whistle']):
    """
    Creates a confusion matrix treating each unique combination of calls as a separate class.
    
    Args:
        results_df: DataFrame with columns like 'ECHO_true', 'ECHO_pred', etc.
        call_types: List of call types to include in the combinations
    
    Returns:
        tuple: (confusion_matrix, labels_dict) where labels_dict maps class indices to descriptive names
    """
    # Create true and predicted combinations
    true_combinations = []
    pred_combinations = []
    
    for _, row in results_df.iterrows():
        # Get true combination
        true_combo = tuple(row[f'{ct}_true'] for ct in call_types)
        true_combinations.append(true_combo)
        
        # Get predicted combination
        pred_combo = tuple(row[f'{ct}_pred'] for ct in call_types)
        pred_combinations.append(pred_combo)
    
    # Get unique combinations and create mapping
    all_combinations = set(true_combinations + pred_combinations)
    
    # Sort combinations by number of calls present (0, 1, 2, etc.)
    sorted_combinations = sorted(all_combinations, key=lambda x: (sum(x), x))
    combo_to_idx = {combo: idx for idx, combo in enumerate(sorted_combinations)}
    
    # Convert combinations to class indices
    y_true = [combo_to_idx[combo] for combo in true_combinations]
    y_pred = [combo_to_idx[combo] for combo in pred_combinations]
    
    # Create confusion matrix
    n_classes = len(combo_to_idx)
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf_matrix[t, p] += 1
    
    # Create labels dictionary
    labels_dict = {}
    for combo, idx in combo_to_idx.items():
        if sum(combo) == 0:
            label = "Absent"
        else:
            present_calls = [call_types[i] for i, val in enumerate(combo) if val == 1]
            label = "+".join(present_calls)
        labels_dict[str(idx)] = label
    
    return conf_matrix, labels_dict

def display_multiclass_confusion_matrix(results_df, call_types=['ECHO', 'HFPC', 'CC', 'Whistle'], figsize=(8, 6)):
    """
    Displays a confusion matrix treating each unique combination of calls as a separate class.
    Each cell shows both the percentage and raw count.
    
    Args:
        results_df: DataFrame with columns like 'ECHO_true', 'ECHO_pred', etc.
        call_types: List of call types to include in the combinations
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create confusion matrix
    conf_matrix, labels_dict = create_multiclass_confusion_matrix(results_df, call_types)
    
    # Calculate percentages
    conf_matrix_percentages = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    # Create labels list in the correct order
    labels = [labels_dict[str(i)] for i in range(len(labels_dict))]
    
    # Create annotation matrix with both percentage and count
    annot_matrix = np.empty_like(conf_matrix, dtype=object)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            annot_matrix[i, j] = f'{conf_matrix_percentages[i, j]:.1%}\n({conf_matrix[i, j]})'
    
    # Plot confusion matrix
    plt.figure(figsize=figsize)
    sns.heatmap(conf_matrix_percentages, annot=annot_matrix, fmt='', cmap='Greens',
                xticklabels=labels, yticklabels=labels, vmin=0, vmax=1, cbar=False)
    # plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # Print classification metrics
    from sklearn.metrics import classification_report
    y_true = [labels_dict[str(i)] for i in range(len(labels_dict)) for _ in range(int(conf_matrix[i].sum()))]
    y_pred = [labels_dict[str(j)] for i in range(len(labels_dict)) for j in range(len(labels_dict)) for _ in range(int(conf_matrix[i, j]))]
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

from spectrogram.spectrogram_generator import SPECT_GENERATOR, HYDROPHONE_SENSITIVITY
import librosa

def plot_combined_spectrograms(audio, sr, title="title", clip_number=None):
    """
    Plot both Mel and linear spectrograms side by side in one figure.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        title: Title for the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    

    roi_df = None
    if clip_number is not None:
        roi_df = pd.DataFrame({
        'min_x_s': [clip_number-1],  # Start at beginning of segment
        'max_x_s': [clip_number],  # End at end of segment
        'min_freq_hz': [0],  # Use frequency if available
        'max_freq_hz': [95000],  # Use frequency if available
        
    })

    # Left plot: Mel spectrogram
    plt.sca(axes[0])
    power_spect = SPECT_GENERATOR.compute_mel_power_spect(audio)
    dB_spect = SPECT_GENERATOR.power_to_db(power_spect)
    SPECT_GENERATOR.plot_spect(dB_spect, ax=axes[0], title="Mel Spectrogram", show=False, rois_df=roi_df)
    
    plt.sca(axes[1])
    n_fft = 1024  # FFT window size
    hop_length = 300  # Hop length (controls overlap)
    S = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)),
        ref=np.max
    )
    
    # Increase contrast by adjusting vmin and vmax parameters
    # This will clip the color range to enhance visibility of features
    vmin = np.max(S) - 80  # Show only the top 80dB range (adjust as needed)
    img = librosa.display.specshow(S, sr=sr, hop_length=hop_length, 
                                  x_axis='time', y_axis='hz', ax=axes[1],
                                  vmin=vmin, vmax=np.max(S),
                                  )  # Try different colormaps like 'magma', 'inferno', 'plasma'
    
    plt.colorbar(img, format='%+2.0f dB', label='Magnitude (dB)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title("Linear Spectrogram")
    
    # Add overall title
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)  # Make room for the suptitle
    
    return fig

from sklearn.metrics import f1_score
def calculate_detection_f1(test_predictions_df):
    # Create call presence columns
    test_predictions_df['call_presence_true'] = (
        (test_predictions_df['ECHO_true'] == 1) | 
        (test_predictions_df['HFPC_true'] == 1) | 
        (test_predictions_df['CC_true'] == 1) | 
        (test_predictions_df['Whistle_true'] == 1)
    ).astype(int)

    test_predictions_df['call_presence_pred'] = (
        (test_predictions_df['ECHO_pred'] == 1) | 
        (test_predictions_df['HFPC_pred'] == 1) | 
        (test_predictions_df['CC_pred'] == 1) | 
        (test_predictions_df['Whistle_pred'] == 1)
    ).astype(int)

    
    if 'Fold' in test_predictions_df.columns:
        fold_f1_scores = []
        for fold in range(5):  # Assuming folds 0-4
            fold_mask = test_predictions_df['Fold'] == fold
            fold_true = test_predictions_df.loc[fold_mask, 'call_presence_true']
            fold_pred = test_predictions_df.loc[fold_mask, 'call_presence_pred']
            f1 = f1_score(fold_true, fold_pred, average='macro')
            fold_f1_scores.append(f1)

        # Create the final result dictionary
        detection_f1 = {
            "values": fold_f1_scores,
            "mean": np.mean(fold_f1_scores),
            "std": np.std(fold_f1_scores)
        }
    else:
        # Calculate single F1 score if no folds
        f1 = f1_score(test_predictions_df['call_presence_true'], 
                     test_predictions_df['call_presence_pred'],
                     average='macro')
        detection_f1 = f1
    return detection_f1