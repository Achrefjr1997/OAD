"""
OAD Data Processing Package

This package provides functionalities for downloading datasets and preparing
remote sensing images for analysis. It includes tools for calculating PCA
and various spectral indices.
"""

# Import necessary functions or classes
from .download import download_files
from .prepare_data import process_images,normalize_sentinel2_image
from .Loss import AoDLoss

__all__ = [
    'download_files',
    'prepare_data',
    'normalize_sentinel2_image',
    'AoDLoss'
]