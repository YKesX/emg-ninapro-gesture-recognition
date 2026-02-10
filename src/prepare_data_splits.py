"""
Data splitting utility for creating persistent train/val/test splits.

This script loads the original training data and creates a proper train/validation/test
split with a fixed random seed to ensure consistency across training and evaluation.
The splits are saved as separate .npy files to prevent train/test overlap.
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split


def create_data_splits(
    x_path='data/processed/X_train.npy',
    y_path='data/processed/y_train.npy',
    val_size=0.2,
    test_size=0.2,
    random_state=42
):
    """
    Create and save persistent train/validation/test splits.
    
    Args:
        x_path: Path to the original X data file
        y_path: Path to the original y data file
        val_size: Proportion of data to use for validation (0.2 = 20%)
        test_size: Proportion of data to use for testing (0.2 = 20%)
        random_state: Random seed for reproducibility
    """
    print("Loading original data...")
    
    # Try to load from different possible locations
    if not os.path.exists(x_path):
        x_path = '../data/processed/X_train.npy'
        y_path = '../data/processed/y_train.npy'
        if not os.path.exists(x_path):
            raise FileNotFoundError(f"Data file not found: {x_path}")
    
    X = np.load(x_path)
    y = np.load(y_path)
    
    print(f"Original data shape: X={X.shape}, y={y.shape}")
    
    # First split: separate test set
    # test_size is the proportion of the whole dataset
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate validation from training
    # val_size needs to be adjusted for the remaining data
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    print(f"Train set: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation set: X={X_val.shape}, y={y_val.shape}")
    print(f"Test set: X={X_test.shape}, y={y_test.shape}")
    
    # Save the splits
    output_dir = os.path.dirname(x_path)
    
    np.save(os.path.join(output_dir, 'X_train_split.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train_split.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_val_split.npy'), X_val)
    np.save(os.path.join(output_dir, 'y_val_split.npy'), y_val)
    np.save(os.path.join(output_dir, 'X_test_split.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test_split.npy'), y_test)
    
    print(f"\nSplits saved to {output_dir}/")
    print("Files created:")
    print("  - X_train_split.npy, y_train_split.npy (training)")
    print("  - X_val_split.npy, y_val_split.npy (validation)")
    print("  - X_test_split.npy, y_test_split.npy (testing)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    create_data_splits()
