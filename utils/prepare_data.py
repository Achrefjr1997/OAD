import os
import numpy as np
import cv2
import pandas as pd
from sklearn.decomposition import PCA
import tifffile as tiff
# Define band information based on the provided table
bands_info = {
    "B1": {"description": "Aerosols", "wavelength": "443.9nm (S2A) / 442.3nm (S2B)"},
    "B2": {"description": "Blue", "wavelength": "496.6nm (S2A) / 492.1nm (S2B)"},
    "B3": {"description": "Green", "wavelength": "560nm (S2A) / 559nm (S2B)"},
    "B4": {"description": "Red", "wavelength": "664.5nm (S2A) / 665nm (S2B)"},
    "B5": {"description": "Red Edge 1", "wavelength": "703.9nm (S2A) / 703.8nm (S2B)"},
    "B6": {"description": "Red Edge 2", "wavelength": "740.2nm (S2A) / 739.1nm (S2B)"},
    "B7": {"description": "Red Edge 3", "wavelength": "782.5nm (S2A) / 779.7nm (S2B)"},
    "B8": {"description": "NIR", "wavelength": "835.1nm (S2A) / 833nm (S2B)"},
    "B8A": {"description": "Red Edge 4", "wavelength": "864.8nm (S2A) / 864nm (S2B)"},
    "B9": {"description": "Water vapor", "wavelength": "945nm (S2A) / 943.2nm (S2B)"},
    "B11": {"description": "SWIR 1", "wavelength": "1613.7nm (S2A) / 1610.4nm (S2B)"},
    "B12": {"description": "SWIR 2", "wavelength": "2202.4nm (S2A) / 2185.7nm (S2B)"},
}

def calculate_pca(image):
    """Calculate PCA for the given image and return the first three components."""
    # Reshape the image to be a 2D array of pixels
    image=np.clip(image, 0, 1)
    pixels = image.reshape(-1, image.shape[2])
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(pixels)
    # Reshape back to the original image dimensions and stack with the original
    pca_image = np.stack([pca_result[:, i].reshape(image.shape[0], image.shape[1]) for i in range(3)], axis=-1)
    return np.concatenate((image, pca_image), axis=-1)

def calculate_indices(image):
    """Calculate spectral indices and stack them with the original image."""
    image=np.clip(image, 0, 1)
    B4 = image[..., 3]  # Red
    B8 = image[..., 7]  # NIR
    B5 = image[..., 4]  # Red Edge 1
    B3 = image[..., 2]  # Green
    
    # NDVI
    ndvi = (B8 - B4) / (B8 + B4 + 1e-10)
    
    # ATSAVI
    atsavi = (B8 - B4) / (B8 + B4 + 0.5)  # Parameter can be tuned
    
    # ARI
    ari = B8 - (B5 + B3) / 2
    
    # BRI
    bri = (B4 + B3) / 2  # Example calculation, may vary based on definition
    
    # Norm NIR
    norm_nir = B8 / np.max(B8)  # Normalization
    
    # Stack the indices with the original image
    indices_stack = np.stack((ndvi, atsavi, ari, bri, norm_nir), axis=-1)
    return np.concatenate((image, indices_stack), axis=-1)

def process_images(image_folder, output_folder, ACP3C=True, include_index=False):
    """Process images in the specified folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(image_folder):
        if filename.endswith('.tif'):  # Adjust the extension as needed
            image_path = os.path.join(image_folder, filename)
            # Load the image
            image = tiff.imread(image_path)
            H, W, C = image.shape
            # Process with PCA if specified
            if ACP3C:
                processed_image = calculate_pca(image)
            elif include_index:
                processed_image = calculate_indices(image)
            else:
                processed_image = image  # No processing
            processed_image=np.transpose(processed_image, (2, 0, 1))

            # Save the processed image
            output_path = os.path.join(output_folder, filename)
            tiff.imwrite(output_path, processed_image.astype(np.float32))

def prepare_data(ACP3C=False, include_index=False):
    # Process the training images
    process_images('Dataset/train_images', 'Processed/train_images', ACP3C=ACP3C, include_index=include_index)
    # Process the test images
    process_images('Dataset/test_images', 'Processed/test_images', ACP3C=ACP3C, include_index=include_index)

    # Load the training answers
    train_answers = pd.read_csv('Dataset/train_answer.csv')
    # Save the training answers with the same format
    train_answers.to_csv('Processed/train_answer.csv', index=False)

def normalize_sentinel2_image(tiff_file_path):
    """
    Normalize Sentinel-2 multi-band TIFF image from 16-bit to 8-bit.

    Parameters:
        tiff_file_path (str): Path to the input TIFF file.
        output_file_path (str): Path to save the normalized 8-bit TIFF file.
    """
    image_data=tiff.imread(tiff_file_path)

    # Normalize the remaining channels (B05, B06, B07, B08, B11, B12)
    for i in range(0, 4):  # For B05 to B12
        image_data[i] = image_data[i]* 1100 / (255*255)
    for i in range(4, 13):  # For B05 to B12
        image_data[i] = image_data[i]* 8160 / (255*255)  # Divide by 8160
    image_data = np.clip(image_data, 0, 1)
    return image_data