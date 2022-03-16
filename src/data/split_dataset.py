"""Split the images in train/test/valid sets."""

import os

import numpy as np
from sklearn.model_selection import train_test_split

EPS = 1e-3


def generate_indices():
    """
    Generate train/test/validation sets indices.
    To be used with "torch.utils.data.Subset".
    Train: 80%
    Test: 15%
    Valid: 5%
    """

    output_dir = "../../data/processed/"

    # 1501 tiles without pools (class 0)
    # 1577 tiles with pools (class 1)
    X = np.arange(1501 + 1577)
    y = np.zeros(1501 + 1577, dtype=np.int)
    y[1501:] = 1
    class_imbalance_ratio = 1577 / (1577 + 1501)

    # X_rem/y_rem: remaining indices (train+valid)
    # X_test/y_test: indices for test set (15% of full dataset)
    X_rem, X_test, y_rem, y_test = train_test_split(X, y, test_size=0.15, stratify=y)
    # X_train/y_train: indices for training set
    # X_valid/y_valid: indices for validation set (5% of full dataset)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_rem, y_rem, test_size=0.05 / (1 - 0.15), stratify=y_rem
    )
    del X_rem, y_rem

    train_indices = X_train
    test_indices = X_test
    valid_indices = X_valid

    for set_name, y_indices in zip(
        ("y_train", "y_test", "y_valid"), (y_train, y_test, y_valid)
    ):
        print(set_name, "length:", len(y_indices))
        print("   class pool: ", y_indices.sum())
        print("   class no pool: ", (1 - y_indices).sum())
        assert (
            y_indices.sum() / (y_indices.sum() + (1 - y_indices).sum())
            - class_imbalance_ratio
        ) < EPS

    assert len(train_indices) + len(test_indices) + len(valid_indices) == 1577 + 1501

    # Check if there's an overlap
    assert not set(train_indices) & set(test_indices)
    assert not set(test_indices) & set(valid_indices)
    assert not set(train_indices) & set(valid_indices)

    # Save indices
    np.save(os.path.join(output_dir, "train_indices.npy"), train_indices)
    np.save(os.path.join(output_dir, "test_indices.npy"), test_indices)
    np.save(os.path.join(output_dir, "valid_indices.npy"), valid_indices)


if __name__ == "__main__":
    generate_indices()
