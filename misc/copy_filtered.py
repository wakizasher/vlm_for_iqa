import os
import shutil
import pandas as pd

# Path to your CSV file
csv_path = r'/data/filtered_iqa.csv'

# Target directory for filtered images
target_dir = r'D:\iNaturalist\images_filtered'

# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_path)

# The CSV column with full file paths is 'file_path'
for src_path in df['file_path']:
    if os.path.isfile(src_path):  # Check if the file exists
        # Copy the file to the target directory, keeping the original filename
        shutil.copy2(src_path, target_dir)
    else:
        print(f"File not found: {src_path}")