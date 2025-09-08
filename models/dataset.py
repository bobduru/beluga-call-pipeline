import random
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, Dataset

class MultiLabelDataset(torch.utils.data.Dataset):
    """
    Read processed audio files (which are now spectrogram images),
    and return them as tensors.
    """
    def __init__(
        self,
        df,
        processed_spects_dir=None,
        label_columns=None,
        train=True,  # Parameter to control augmentation
        spect_filename_column='Filename',
    ): 
        self.df = df
        self.processed_files_dir = processed_spects_dir
        # Default mapping if none provided (for backward compatibility)
        
        self.label_columns = label_columns
        self.train = train  # Determines if augmentation is applied
        self.spect_filename_column = spect_filename_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row[self.spect_filename_column]
        # print(row[self.label_columns].values)
        labels = torch.tensor(row[self.label_columns].astype(float).values, dtype=torch.float32)
        # Load spectrogram
        file_path = self.processed_files_dir + filename
        spect = torch.load(file_path, weights_only=False)
        spect = torch.from_numpy(spect)
        spect = spect.unsqueeze(0)  # Add channel dimension

        # Metadata stays the same
        metadata = {"Site": row["Site"], "Filename": filename}

        if "N_HF_Call_Types" in row:
            metadata["N_HF_Call_Types"] = row["N_HF_Call_Types"]

        return spect, labels, metadata
    

def setup_multilabel_dataloaders(train_data, val_data, test_data, processed_spects_dir, label_columns, batch_size=32):
    train_dataset = MultiLabelDataset(
        train_data,
        processed_spects_dir=processed_spects_dir,
        label_columns=label_columns,
        train=True,
        spect_filename_column='ClipFilenamePt'
    )
    val_dataset = MultiLabelDataset(
        val_data,
        processed_spects_dir=processed_spects_dir,
       label_columns=label_columns,
        spect_filename_column='ClipFilenamePt'

    )
    test_dataset = MultiLabelDataset(
        test_data,
        processed_spects_dir=processed_spects_dir,
        label_columns=label_columns,
        spect_filename_column='ClipFilenamePt'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader