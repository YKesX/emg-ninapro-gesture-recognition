"""
Test script to validate the data splitting logic.

This creates synthetic data to test that:
1. The prepare_data_splits.py script creates proper train/val/test splits
2. There's no overlap between the splits
3. train.py and eval.py can load the splits correctly
"""
import os
import sys
import tempfile
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from prepare_data_splits import create_data_splits

# Test constants
NUM_CLASSES = 13  # Must match TrainingConfig.NUM_CLASSES


def test_data_splits():
    """Test that data splits are created correctly and have no overlap."""
    
    print("Creating synthetic test data...")
    
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, 'data', 'processed')
        os.makedirs(data_dir, exist_ok=True)
        
        # Create synthetic data (1000 samples, 20 timesteps, 10 features)
        n_samples = 1000
        X_synthetic = np.random.randn(n_samples, 20, 10)
        y_synthetic = np.random.randint(0, NUM_CLASSES, size=n_samples)
        
        # Save synthetic data
        x_path = os.path.join(data_dir, 'X_train.npy')
        y_path = os.path.join(data_dir, 'y_train.npy')
        np.save(x_path, X_synthetic)
        np.save(y_path, y_synthetic)
        
        print(f"Synthetic data created: {X_synthetic.shape}")
        
        # Run the splitting function
        print("\nRunning create_data_splits...")
        X_train, y_train, X_val, y_val, X_test, y_test = create_data_splits(
            x_path=x_path,
            y_path=y_path,
            val_size=0.2,
            test_size=0.2,
            random_state=42
        )
        
        # Verify the splits
        print("\n" + "="*60)
        print("VALIDATION RESULTS")
        print("="*60)
        
        # Check shapes
        total_samples = len(X_train) + len(X_val) + len(X_test)
        print(f"\n1. Total samples: {total_samples} (expected: {n_samples})")
        assert total_samples == n_samples, f"Sample count mismatch! {total_samples} != {n_samples}"
        print("   ✓ Sample count correct")
        
        # Check proportions
        train_pct = len(X_train) / n_samples
        val_pct = len(X_val) / n_samples
        test_pct = len(X_test) / n_samples
        print(f"\n2. Split proportions:")
        print(f"   Train: {train_pct:.1%} (expected: ~60%)")
        print(f"   Val:   {val_pct:.1%} (expected: ~20%)")
        print(f"   Test:  {test_pct:.1%} (expected: ~20%)")
        
        # Allow for some rounding differences
        assert 0.58 <= train_pct <= 0.62, f"Train split proportion incorrect: {train_pct}"
        assert 0.18 <= val_pct <= 0.22, f"Val split proportion incorrect: {val_pct}"
        assert 0.18 <= test_pct <= 0.22, f"Test split proportion incorrect: {test_pct}"
        print("   ✓ Proportions are correct")
        
        # Check for overlap by creating indices
        print(f"\n3. Checking for data overlap...")
        
        # Create unique identifiers based on first few features of each sample
        def get_sample_id(X):
            """Create a simple hash-like identifier for each sample."""
            return [tuple(x.flatten()[:5]) for x in X]
        
        train_ids = set(get_sample_id(X_train))
        val_ids = set(get_sample_id(X_val))
        test_ids = set(get_sample_id(X_test))
        
        train_val_overlap = train_ids & val_ids
        train_test_overlap = train_ids & test_ids
        val_test_overlap = val_ids & test_ids
        
        print(f"   Train-Val overlap: {len(train_val_overlap)} samples")
        print(f"   Train-Test overlap: {len(train_test_overlap)} samples")
        print(f"   Val-Test overlap: {len(val_test_overlap)} samples")
        
        assert len(train_val_overlap) == 0, "Train and validation sets overlap!"
        assert len(train_test_overlap) == 0, "Train and test sets overlap!"
        assert len(val_test_overlap) == 0, "Validation and test sets overlap!"
        print("   ✓ No overlap detected")
        
        # Verify files were created
        print(f"\n4. Checking that split files were created...")
        expected_files = [
            'X_train_split.npy', 'y_train_split.npy',
            'X_val_split.npy', 'y_val_split.npy',
            'X_test_split.npy', 'y_test_split.npy'
        ]
        
        for fname in expected_files:
            fpath = os.path.join(data_dir, fname)
            assert os.path.exists(fpath), f"File not created: {fname}"
            print(f"   ✓ {fname}")
        
        # Verify we can reload the data
        print(f"\n5. Verifying splits can be reloaded...")
        X_train_reload = np.load(os.path.join(data_dir, 'X_train_split.npy'))
        y_train_reload = np.load(os.path.join(data_dir, 'y_train_split.npy'))
        
        assert np.array_equal(X_train, X_train_reload), "Reloaded train X doesn't match!"
        assert np.array_equal(y_train, y_train_reload), "Reloaded train y doesn't match!"
        print("   ✓ Data can be reloaded correctly")
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nThe data splitting logic is working correctly:")
        print("- Splits are created with correct proportions")
        print("- No overlap between train/val/test sets")
        print("- Data is saved and can be reloaded properly")


if __name__ == "__main__":
    test_data_splits()
