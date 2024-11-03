"""
OAD Data Processing Package

This package provides functionalities for downloading datasets and preparing
remote sensing images for analysis. It includes tools for calculating PCA
and various spectral indices.
"""

# Import necessary functions or classes
from .dataloader import transformer,OaDataset,list_checkpoints,OaDPrediction
from .model import OADModel
__all__ = [
    'transformer',
    'OaDataset',
    'OADModel',
    'OaDPrediction',
    'list_checkpoints'
]