import os
import requests
import zipfile

def download_file(url, output_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for bad responses
    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def download_files():
    output_directory = "Dataset"

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Define the list of files to check and their corresponding URLs
    files_info = {
        "train_answer.csv": "https://solafune-dev-v1.s3.us-west-2.amazonaws.com/competitions/aod_estimation/train_answer.csv",
        "sample_answer.csv": "https://solafune-dev-v1.s3.us-west-2.amazonaws.com/competitions/aod_estimation/sample_answer.csv",
        "test_images.zip": "https://solafune-dev-v1.s3.us-west-2.amazonaws.com/competitions/aod_estimation/test_images.zip",
        "train_images.zip": "https://solafune-dev-v1.s3.us-west-2.amazonaws.com/competitions/aod_estimation/train_images.zip"
    }

    # Check for existence of files and download if they do not exist
    for filename, url in files_info.items():
        output_path = os.path.join(output_directory, filename)

        if not os.path.exists(output_path):
            print(f"{filename} not found. Downloading...")
            download_file(url, output_path)
            print(f"Downloaded to {output_path}")
        else:
            print(f"{filename} already exists.")

    # Extract the ZIP files
    zip_files = {
        "test_images.zip": "",
        "train_images.zip": ""
    }

    for zip_filename, extract_folder in zip_files.items():
        zip_path = os.path.join(output_directory, zip_filename)
        extract_to = os.path.join(output_directory, extract_folder)

        if os.path.exists(zip_path):
            # Create the extraction directory if it doesn't exist
            os.makedirs(extract_to, exist_ok=True)
            print(f"Extracting {zip_filename} to {extract_to}...")
            extract_zip(zip_path, extract_to)
            print(f"Extracted to {extract_to}")

if __name__ == "__main__":
    download_files()
