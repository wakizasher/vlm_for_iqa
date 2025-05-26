import os
import shutil
import random
import re


def extract_flower_samples(source_folder, destination_folder, flower_types, samples_per_type=1000):
    """
    Extract a specific number of random samples for each flower type and copy to destination folder.

    Args:
        source_folder: Path to the folder containing all flower images
        destination_folder: Path to the folder where samples will be copied
        flower_types: List of flower names to extract
        samples_per_type: Number of samples to extract per flower type
    """
    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created destination folder: {destination_folder}")

    # Dictionary to store the list of files for each flower type
    flower_files = {flower: [] for flower in flower_types}

    # Supported image extensions
    valid_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

    print(f"Scanning files in {source_folder}...")

    # Collect all matching files for each flower type
    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)

        # Check if it's a file and has a valid image extension
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext.lower()) for ext in valid_extensions):
            # Check if the filename matches one of the flower patterns
            for flower in flower_types:
                # Using regex to match {flowername}_{number}.ext pattern
                if re.match(f"^{flower}_\\d+\\.", filename):
                    flower_files[flower].append(filename)
                    break

    # Copy random samples for each flower type
    for flower, files in flower_files.items():
        print(f"Found {len(files)} images for {flower}")

        # If we don't have enough images for this flower
        if len(files) < samples_per_type:
            print(f"Warning: Only {len(files)} images available for {flower}, copying all of them")
            sample_files = files
        else:
            # Take random samples
            sample_files = random.sample(files, samples_per_type)

        # Create a subfolder for this flower type
        flower_dest_folder = os.path.join(destination_folder, flower)
        if not os.path.exists(flower_dest_folder):
            os.makedirs(flower_dest_folder)
            print(f"Created subfolder: {flower_dest_folder}")

        # Copy the files
        for i, filename in enumerate(sample_files):
            src_path = os.path.join(source_folder, filename)
            dest_path = os.path.join(flower_dest_folder, filename)
            shutil.copy2(src_path, dest_path)
            if (i + 1) % 100 == 0 or i == len(sample_files) - 1:
                print(f"Copied {i + 1}/{len(sample_files)} images for {flower}")

        print(f"Completed copying {len(sample_files)} images for {flower}")


if __name__ == "__main__":
    # Define the flower types
    flower_types = ["Bellis_perennis", "Leucanthemum_vulgare", "Matricaria_chamomilla"]

    # Define the source and destination folders on the external drive
    source_folder = r"D:\iNaturalist\images"
    destination_folder = r"D:\iNaturalist\test_3000"

    print(f"Starting extraction of flower samples:")
    print(f"Source: {source_folder}")
    print(f"Destination: {destination_folder}")
    print(f"Flower types: {', '.join(flower_types)}")
    print(f"Samples per type: 1000")

    # Extract the samples
    extract_flower_samples(source_folder, destination_folder, flower_types)

    print(f"Process completed. Extracted samples are available at: {destination_folder}")