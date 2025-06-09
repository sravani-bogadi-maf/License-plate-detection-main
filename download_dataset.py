import os
import urllib.request
import zipfile

# Check if the dataset zip file exists
zip_file = "License Plate Recognition.v11i.yolov9.zip"
if not os.path.exists(zip_file):
    print(f"Downloading {zip_file}...")
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/license_plate_dataset.zip"
    urllib.request.urlretrieve(url, zip_file)
    print("Download complete!")
else:
    print(f"{zip_file} already exists.")

# Extract the zip file
if os.path.exists(zip_file):
    print(f"Extracting {zip_file}...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Extraction complete!")
    
    # Verify the extracted directories
    if os.path.exists("train") and os.path.exists("valid"):
        print("Dataset extracted successfully!")
        print(f"Train images: {len(os.listdir('train/images'))}")
        print(f"Valid images: {len(os.listdir('valid/images'))}")
    else:
        print("Error: Dataset extraction failed!")
else:
    print(f"Error: {zip_file} not found!")