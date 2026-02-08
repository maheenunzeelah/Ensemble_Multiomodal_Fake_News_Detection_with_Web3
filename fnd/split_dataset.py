import torch
from torch.utils.data import Dataset, random_split, DataLoader
import pandas as pd
from typing import Tuple, Optional, List
from pathlib import Path


class CSVDataset(Dataset):
    """Custom Dataset for CSV data."""
    
    def __init__(self, csv_path: str, transform=None):
        """
        Args:
            csv_path: Path to the CSV file
            transform: Optional transform to apply to data
        """
        self.data = pd.read_csv(csv_path)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        if self.transform:
            row = self.transform(row)
        
        return row


def split_csv_torch(
    csv_path: str,
    train_ratio: float = 0.8,
    seed: Optional[int] = 42
) -> Tuple[Dataset, Dataset]:
    """
    Split CSV dataset using PyTorch's random_split.
    
    Args:
        csv_path: Path to CSV file
        train_ratio: Ratio of data for training (default: 0.8)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    dataset = CSVDataset(csv_path)
    
    # Calculate lengths
    train_len = int(len(dataset) * train_ratio)
    test_len = len(dataset) - train_len
    
    # Split dataset
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])
    
    return train_dataset, test_dataset


def split_csv_to_files(
    csv_path: str,
    train_path: str,
    test_path: str,
    train_ratio: float = 0.8,
    seed: Optional[int] = 42,
    shuffle: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split CSV file and save to separate train/test CSV files.
    
    Args:
        csv_path: Path to input CSV file
        train_path: Path to save training CSV
        test_path: Path to save test CSV
        train_ratio: Ratio of data for training (default: 0.8)
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle data before splitting
    
    Returns:
        Tuple of (train_df, test_df)
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Shuffle if requested
    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Calculate split index
    split_idx = int(len(df) * train_ratio)
    
    # Split dataframe
    train_df = df[:split_idx]
    test_df = df[split_idx:]
    
    # Save to files
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Train set: {len(train_df)} rows saved to {train_path}")
    print(f"Test set: {len(test_df)} rows saved to {test_path}")
    
    return train_df, test_df


def split_csv_stratified(
    csv_path: str,
    train_path: str,
    test_path: str,
    target_column: str,
    train_ratio: float = 0.8,
    seed: Optional[int] = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split CSV with stratification (maintains class distribution).
    
    Args:
        csv_path: Path to input CSV file
        train_path: Path to save training CSV
        test_path: Path to save test CSV
        target_column: Column name to stratify by
        train_ratio: Ratio of data for training (default: 0.8)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(csv_path)
    
    # Stratified split
    train_df, test_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=df[target_column],
        random_state=seed
    )
    
    # Save to files
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Stratified split by '{target_column}':")
    print(f"Train set: {len(train_df)} rows")
    print(f"Test set: {len(test_df)} rows")
    print(f"\nClass distribution in train:")
    print(train_df[target_column].value_counts(normalize=True))
    
    return train_df, test_df


def create_dataloaders_from_csv(
    csv_path: str,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    seed: Optional[int] = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders from CSV file.
    
    Args:
        csv_path: Path to CSV file
        batch_size: Batch size for DataLoader
        train_ratio: Ratio of data for training
        seed: Random seed for reproducibility
        num_workers: Number of worker processes for data loading
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_dataset, test_dataset = split_csv_torch(csv_path, train_ratio, seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader


# Example usage
if __name__ == "__main__":
    print("\nExample 2: Balance then stratified split")
    
    from sklearn.utils import resample
    
    # Load dataset
    df = pd.read_csv("datasets/processed/final_alldata_dataset_with_entities.csv")
    print(f"Original dataset size: {len(df)}")
    print("Class distribution (BEFORE balancing):")
    print(df['type'].value_counts())
    
    # Separate by label
    fake_samples = df[df['type'] == 'fake']
    real_samples = df[df['type'] == 'real']
    
    # Balance to minimum class size
    min_samples = min(len(fake_samples), len(real_samples))
    
    fake_balanced = resample(fake_samples, n_samples=min_samples, random_state=42)
    real_balanced = resample(real_samples, n_samples=min_samples, random_state=42)
    
    # Combine
    balanced_df = pd.concat([fake_balanced, real_balanced], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nBalanced dataset size: {len(balanced_df)}")
    print("Class distribution (AFTER balancing):")
    print(balanced_df['type'].value_counts())
    
    # Save balanced dataset
    balanced_df.to_csv("datasets/processed/final_alldata_dataset_balanced.csv", index=False)
    
    # Stratified split on balanced data
    train_df, test_df = split_csv_stratified(
        csv_path="datasets/processed/final_alldata_dataset_balanced.csv",
        train_path="all_data_train_balanced.csv",
        test_path="all_data_test_balanced.csv",
        target_column='type',
        train_ratio=0.75,
        seed=42
    )
    
    print(f"\n{'='*70}")
    print("FINAL TRAIN SET:")
    print(train_df['type'].value_counts())
    print("\n{'='*70}")
    print("FINAL TEST SET:")
    print(test_df['type'].value_counts())