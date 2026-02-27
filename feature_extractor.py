"""
Feature Extractor for PDFG (Palmprint Data and Feature Generation)
Architecture matches Fig. 3 from the paper:
  - Shared layers: Conv1→MaxPool→Conv2→MaxPool→Conv3→Conv4→MaxPool
  - Specific layers: FC1→FC2→FC3 (per dataset)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedLayers(nn.Module):
    """Shared convolutional backbone used across all N feature extractors."""

    def __init__(self):
        super().__init__()
        # Conv1: 3×3×16, stride 4, Leaky ReLU
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=4, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)

        # Conv2: 5×5×32, stride 2, Leaky ReLU
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)

        # Conv3: 3×3×64, stride 1, Leaky ReLU
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Conv4: 3×3×128, stride 1, Leaky ReLU
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.pool1(x)
        x = self.leaky_relu(self.conv2(x))
        x = self.pool2(x)
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = self.pool3(x)
        return x


class SpecificLayers(nn.Module):
    """Dataset-specific fully connected layers (one set per source dataset)."""

    def __init__(self, input_dim: int, feature_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, feature_dim)  # Output: 128-dim feature
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)             # No activation on final layer
        return x


class FeatureExtractor(nn.Module):
    """
    Complete feature extractor for one dataset.
    Combines shared layers + dataset-specific layers.
    """

    def __init__(self, input_size: int = 112, feature_dim: int = 128):
        super().__init__()
        self.shared = SharedLayers()
        # Compute flattened size after shared layers
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_size, input_size)
            out = self.shared(dummy)
            flat_dim = out.view(1, -1).shape[1]
        self.specific = SpecificLayers(flat_dim, feature_dim)

    def forward(self, x):
        x = self.shared(x)
        x = self.specific(x)
        # L2 normalize output features
        x = F.normalize(x, p=2, dim=1)
        return x


class MultiDatasetExtractors(nn.Module):
    """
    N feature extractors sharing the same convolutional layers
    but having separate fully connected (specific) layers — one per dataset.

    Args:
        n_datasets:  Number of source datasets (N in the paper)
        input_size:  Spatial size of input ROI (112 for both databases)
        feature_dim: Output feature dimensionality (128 by default)
    """

    def __init__(self, n_datasets: int, input_size: int = 112, feature_dim: int = 128):
        super().__init__()
        self.n_datasets = n_datasets

        # Single shared backbone
        self.shared = SharedLayers()

        # Compute flattened dimension once
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_size, input_size)
            out = self.shared(dummy)
            flat_dim = out.view(1, -1).shape[1]

        # N dataset-specific FC heads
        self.specific_layers = nn.ModuleList([
            SpecificLayers(flat_dim, feature_dim) for _ in range(n_datasets)
        ])

    def extract(self, x: torch.Tensor, dataset_idx: int) -> torch.Tensor:
        """Extract features using the head for dataset_idx."""
        shared_feat = self.shared(x)
        feat = self.specific_layers[dataset_idx](shared_feat)
        return F.normalize(feat, p=2, dim=1)

    def extract_all(self, x: torch.Tensor) -> list:
        """Extract features using ALL dataset heads (for augmented images)."""
        shared_feat = self.shared(x)
        feats = []
        for head in self.specific_layers:
            f = head(shared_feat)
            feats.append(F.normalize(f, p=2, dim=1))
        return feats

    def forward(self, x: torch.Tensor, dataset_idx: int) -> torch.Tensor:
        return self.extract(x, dataset_idx)
