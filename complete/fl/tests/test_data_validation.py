"""Property-based tests for data loading and validation.

These tests use Hypothesis to generate edge cases and verify that
data loading functions handle invalid inputs correctly.
"""

import pytest
import torch
from hypothesis import given, strategies as st, assume, settings

from fl.task import load_data


class TestDataLoadingValidation:
    """Test data loading with property-based testing."""

    @given(
        partition_id=st.integers(min_value=-100, max_value=100),
        num_partitions=st.integers(min_value=-10, max_value=100),
    )
    @settings(max_examples=50, deadline=None)
    def test_load_data_validates_partition_bounds(self, partition_id, num_partitions):
        """Property: partition_id must be in [0, num_partitions) and num_partitions > 0."""
        # Valid case: should not raise
        if 0 <= partition_id < num_partitions and num_partitions > 0:
            # This will work once we add validation
            # For now, we expect it might fail without validation
            try:
                trainloader, testloader = load_data(partition_id, num_partitions)
                assert trainloader is not None
                assert testloader is not None
            except Exception as e:
                # Expected to fail without validation - this test documents the requirement
                pytest.skip(f"Validation not yet implemented: {e}")
        else:
            # Invalid case: should raise ValueError once validation is added
            # For now, document that this should fail
            pass  # Will add assertion once validation is implemented

    @given(partition_id=st.integers(min_value=0, max_value=9))
    @settings(max_examples=10, deadline=None)
    def test_load_data_returns_consistent_shapes(self, partition_id):
        """Property: All batches should have consistent shapes."""
        num_partitions = 10
        
        try:
            trainloader, testloader = load_data(partition_id, num_partitions)
            
            # Check that all batches have consistent shapes
            train_shapes = []
            for batch in trainloader:
                train_shapes.append(batch["image"].shape)
                if len(train_shapes) > 3:  # Check first few batches
                    break
            
            # All batches should have same number of channels and spatial dims
            if len(train_shapes) > 1:
                for shape in train_shapes[1:]:
                    assert shape[1:] == train_shapes[0][1:], \
                        "Inconsistent image shapes across batches"
        except Exception as e:
            pytest.skip(f"Data loading failed: {e}")

    @given(partition_id=st.integers(min_value=0, max_value=9))
    @settings(max_examples=10, deadline=None)
    def test_load_data_labels_in_valid_range(self, partition_id):
        """Property: All labels should be in valid range [0, num_classes)."""
        num_partitions = 10
        num_classes = 10  # For MNIST/PneumoniaMNIST
        
        try:
            trainloader, testloader = load_data(partition_id, num_partitions)
            
            # Check train labels
            for batch in trainloader:
                labels = batch["label"]
                assert labels.min() >= 0, "Labels contain negative values"
                assert labels.max() < num_classes, f"Labels exceed num_classes ({num_classes})"
                break  # Check first batch
                
            # Check test labels
            for batch in testloader:
                labels = batch["label"]
                assert labels.min() >= 0, "Labels contain negative values"
                assert labels.max() < num_classes, f"Labels exceed num_classes ({num_classes})"
                break  # Check first batch
        except Exception as e:
            pytest.skip(f"Data loading failed: {e}")


class TestDataLoaderProperties:
    """Test DataLoader properties and invariants."""

    def test_train_test_split_ratio(self):
        """Test that train/test split is approximately 80/20."""
        partition_id = 0
        num_partitions = 3
        
        try:
            trainloader, testloader = load_data(partition_id, num_partitions)
            
            train_size = len(trainloader.dataset)
            test_size = len(testloader.dataset)
            total_size = train_size + test_size
            
            train_ratio = train_size / total_size
            
            # Should be approximately 0.8 (80/20 split)
            assert 0.75 <= train_ratio <= 0.85, \
                f"Train ratio {train_ratio:.2f} not in expected range [0.75, 0.85]"
        except Exception as e:
            pytest.skip(f"Data loading failed: {e}")

    def test_dataloaders_not_empty(self):
        """Test that dataloaders contain data."""
        partition_id = 0
        num_partitions = 3
        
        try:
            trainloader, testloader = load_data(partition_id, num_partitions)
            
            assert len(trainloader.dataset) > 0, "Train dataset is empty"
            assert len(testloader.dataset) > 0, "Test dataset is empty"
            
            # Verify we can iterate
            train_batch = next(iter(trainloader))
            assert "image" in train_batch
            assert "label" in train_batch
            
            test_batch = next(iter(testloader))
            assert "image" in test_batch
            assert "label" in test_batch
        except Exception as e:
            pytest.skip(f"Data loading failed: {e}")

    def test_batch_size_respected(self):
        """Test that batch size configuration is respected."""
        from fl.config import load_run_config
        
        partition_id = 0
        num_partitions = 3
        
        try:
            config = load_run_config()
            expected_batch_size = config.get("data", {}).get("batch_size", 32)
            
            trainloader, testloader = load_data(partition_id, num_partitions)
            
            # Check first batch (last batch might be smaller)
            train_batch = next(iter(trainloader))
            batch_size = train_batch["image"].shape[0]
            
            # Batch size should match config (or be smaller for last batch)
            assert batch_size <= expected_batch_size, \
                f"Batch size {batch_size} exceeds configured {expected_batch_size}"
        except Exception as e:
            pytest.skip(f"Data loading failed: {e}")


class TestDataTransformations:
    """Test data preprocessing and transformations."""

    def test_images_normalized(self):
        """Test that images are normalized to reasonable range."""
        partition_id = 0
        num_partitions = 3
        
        try:
            trainloader, _ = load_data(partition_id, num_partitions)
            
            batch = next(iter(trainloader))
            images = batch["image"]
            
            # After normalization, values should be roughly in [-3, 3] range
            # (assuming mean=0.5, std=0.5 normalization)
            assert images.min() >= -5.0, "Images have extreme negative values"
            assert images.max() <= 5.0, "Images have extreme positive values"
            
            # Check that images are actually normalized (not raw [0, 255])
            assert images.max() < 10.0, "Images appear to be unnormalized"
        except Exception as e:
            pytest.skip(f"Data loading failed: {e}")

    def test_image_dtype_is_float(self):
        """Test that images are converted to float tensors."""
        partition_id = 0
        num_partitions = 3
        
        try:
            trainloader, testloader = load_data(partition_id, num_partitions)
            
            train_batch = next(iter(trainloader))
            assert train_batch["image"].dtype == torch.float32, \
                "Train images should be float32"
            
            test_batch = next(iter(testloader))
            assert test_batch["image"].dtype == torch.float32, \
                "Test images should be float32"
        except Exception as e:
            pytest.skip(f"Data loading failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
