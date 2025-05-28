import os
import shutil
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple
import json
from datetime import datetime


def set_reproducible_seed(seed_value: int = 42):
    """
    Set the random seed for reproducible results.

    Args:
        seed_value: The seed value to use (default: 42)

    Why this matters:
    - Ensures same random samples every time
    - Makes your research reproducible
    - Required for scientific validity
    """
    random.seed(seed_value)
    print(f"🌱 Random seed set to {seed_value} for reproducible sampling")
    return seed_value


def extract_flower_samples_single_seed(source_folder: str,
                                       destination_folder: str,
                                       flower_types: List[str],
                                       samples_per_type: int = 200,
                                       seed: int = 42) -> Dict[str, int]:
    """
    Extract samples for a single seed/directory.

    Args:
        source_folder: Path to the folder containing all flower images
        destination_folder: Path to the folder where samples will be copied
        flower_types: List of flower names to extract
        samples_per_type: Number of samples to extract per flower type
        seed: Random seed for reproducible sampling

    Returns:
        Dictionary with statistics about processed files
    """

    # Set seed for reproducibility
    actual_seed = set_reproducible_seed(seed)

    # Create destination folder if it doesn't exist
    dest_path = Path(destination_folder)
    dest_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Destination folder ready: {destination_folder}")

    # Dictionary to store the list of files for each flower type
    flower_files = {flower: [] for flower in flower_types}

    # Statistics tracking
    stats = {
        'total_files_found': 0,
        'files_per_flower': {},
        'samples_copied': {},
        'seed_used': actual_seed,
        'timestamp': datetime.now().isoformat()
    }

    # Supported image extensions
    valid_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

    print(f"🔍 Scanning files in {source_folder}...")

    # Verify source folder exists
    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Source folder not found: {source_folder}")

    # Collect all matching files for each flower type
    total_files_scanned = 0
    for filename in os.listdir(source_folder):
        total_files_scanned += 1
        file_path = os.path.join(source_folder, filename)

        # Check if it's a file and has a valid image extension
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext.lower()) for ext in valid_extensions):
            stats['total_files_found'] += 1

            # Check if the filename matches one of the flower patterns
            for flower in flower_types:
                # Using regex to match {flowername}_{number}.ext pattern
                if re.match(f"^{flower}_\\d+\\.", filename):
                    flower_files[flower].append(filename)
                    break

    # Copy random samples for each flower type
    for flower, files in flower_files.items():
        print(f"🌸 Processing {flower}:")
        print(f"   Found {len(files)} images")
        stats['files_per_flower'][flower] = len(files)

        # If we don't have enough images for this flower
        if len(files) < samples_per_type:
            print(f"   ⚠️  Warning: Only {len(files)} images available, copying all of them")
            sample_files = files.copy()
        else:
            # Set seed for this specific flower to ensure reproducibility
            flower_seed = seed + hash(flower) % 1000
            random.seed(flower_seed)

            # Sort files first to ensure consistent ordering across different systems
            sorted_files = sorted(files)
            sample_files = random.sample(sorted_files, samples_per_type)
            print(f"   🎲 Randomly selected {len(sample_files)} images (seed: {flower_seed})")

        # Create a subfolder for this flower type
        flower_dest_folder = dest_path / flower
        flower_dest_folder.mkdir(exist_ok=True)

        # Copy the files
        copied_count = 0
        for i, filename in enumerate(sample_files):
            src_path = os.path.join(source_folder, filename)
            dest_path_file = flower_dest_folder / filename

            try:
                shutil.copy2(src_path, dest_path_file)
                copied_count += 1

                # Progress reporting every 50 files (since we're dealing with smaller numbers)
                if (i + 1) % 50 == 0 or i == len(sample_files) - 1:
                    print(f"   📋 Copied {i + 1}/{len(sample_files)} images")

            except Exception as e:
                print(f"   ❌ Error copying {filename}: {e}")

        stats['samples_copied'][flower] = copied_count
        print(f"   ✅ Completed: {copied_count} images copied for {flower}")

    return stats


def extract_multiple_seed_datasets(source_folder: str,
                                   base_destination_folder: str,
                                   flower_types: List[str],
                                   seeds: List[int],
                                   samples_per_type: int = 200) -> Dict[int, Dict]:
    """
    Create multiple datasets with different seeds.

    Args:
        source_folder: Path to the folder containing all flower images
        base_destination_folder: Base path where seed directories will be created
        flower_types: List of flower names to extract
        seeds: List of seeds to use for creating different datasets
        samples_per_type: Number of samples to extract per flower type per seed

    Returns:
        Dictionary mapping seed to statistics for that dataset
    """

    all_stats = {}
    base_path = Path(base_destination_folder)
    base_path.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Creating {len(seeds)} datasets with {samples_per_type} samples per flower type")
    print(f"Seeds: {seeds}")
    print(f"Flower types: {', '.join(flower_types)}")
    print("=" * 60)

    for i, seed in enumerate(seeds, 1):
        print(f"\n📂 Creating Dataset {i}/{len(seeds)} with seed {seed}")
        print("-" * 40)

        # Create seed-specific directory
        seed_dir = base_path / f"seed_{seed}"

        try:
            # Extract samples for this seed
            stats = extract_flower_samples_single_seed(
                source_folder=source_folder,
                destination_folder=str(seed_dir),
                flower_types=flower_types,
                samples_per_type=samples_per_type,
                seed=seed
            )

            # Save metadata for this seed
            save_sampling_metadata(str(seed_dir), stats, flower_types, samples_per_type, seed)

            all_stats[seed] = stats

            print(f"✅ Dataset {i} completed successfully!")
            print(f"   Total images copied: {sum(stats['samples_copied'].values())}")

        except Exception as e:
            print(f"❌ Error creating dataset with seed {seed}: {e}")
            all_stats[seed] = {'error': str(e)}

    # Save overall summary
    save_overall_summary(base_destination_folder, all_stats, seeds, flower_types, samples_per_type)

    return all_stats


def save_sampling_metadata(destination_folder: str,
                           stats: Dict,
                           flower_types: List[str],
                           samples_per_type: int,
                           seed: int):
    """
    Save metadata about the sampling process for reproducibility.
    """
    metadata = {
        'sampling_config': {
            'seed_used': seed,
            'flower_types': flower_types,
            'samples_per_type': samples_per_type
        },
        'results': {
            'files_found_per_flower': stats['files_per_flower'],
            'samples_copied_per_flower': stats['samples_copied'],
            'total_files_found': stats['total_files_found'],
            'total_samples_copied': sum(stats['samples_copied'].values())
        },
        'metadata': {
            'timestamp': stats['timestamp'],
            'script_version': '3.0_multiple_seeds'
        }
    }

    metadata_file = Path(destination_folder) / 'sampling_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def save_overall_summary(base_destination_folder: str,
                         all_stats: Dict[int, Dict],
                         seeds: List[int],
                         flower_types: List[str],
                         samples_per_type: int):
    """
    Save an overall summary of all datasets created.
    """
    summary = {
        'experiment_config': {
            'seeds_used': seeds,
            'flower_types': flower_types,
            'samples_per_type': samples_per_type,
            'total_datasets': len(seeds)
        },
        'results_summary': {},
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'script_version': '3.0_multiple_seeds'
        }
    }

    # Summarize results for each seed
    for seed in seeds:
        if seed in all_stats and 'error' not in all_stats[seed]:
            stats = all_stats[seed]
            summary['results_summary'][f'seed_{seed}'] = {
                'total_copied': sum(stats['samples_copied'].values()),
                'per_flower': stats['samples_copied'],
                'status': 'success'
            }
        else:
            summary['results_summary'][f'seed_{seed}'] = {
                'status': 'failed',
                'error': all_stats.get(seed, {}).get('error', 'Unknown error')
            }

    summary_file = Path(base_destination_folder) / 'experiment_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n📋 Overall experiment summary saved to: {summary_file}")


def verify_all_datasets(base_destination_folder: str):
    """
    Verify and display information about all created datasets.
    """
    base_path = Path(base_destination_folder)
    summary_file = base_path / 'experiment_summary.json'

    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)

        print("\n🔍 Dataset Creation Summary:")
        print(f"   Total datasets: {summary['experiment_config']['total_datasets']}")
        print(f"   Seeds used: {summary['experiment_config']['seeds_used']}")
        print(f"   Samples per type: {summary['experiment_config']['samples_per_type']}")
        print(f"   Flower types: {', '.join(summary['experiment_config']['flower_types'])}")

        print("\n📊 Results per dataset:")
        for seed_name, results in summary['results_summary'].items():
            if results['status'] == 'success':
                print(f"   {seed_name}: {results['total_copied']} images ✅")
                for flower, count in results['per_flower'].items():
                    print(f"      {flower}: {count}")
            else:
                print(f"   {seed_name}: Failed ❌ ({results.get('error', 'Unknown error')})")
    else:
        print("❌ No experiment summary found!")


if __name__ == "__main__":
    # Define the flower types
    flower_types = ["Bellis_perennis", "Leucanthemum_vulgare", "Matricaria_chamomilla"]

    # Define the source and base destination folders
    source_folder = r"D:\iNaturalist\images"
    base_destination_folder = r"D:\iNaturalist\test_images_200"

    # Define the seeds for creating different datasets
    SEEDS = [42, 123, 456]  # 🌱 Three different seeds for three datasets
    SAMPLES_PER_TYPE = 200  # 📊 200 samples per flower type

    print(f"🚀 Starting creation of multiple flower sample datasets:")
    print(f"   Source: {source_folder}")
    print(f"   Base destination: {base_destination_folder}")
    print(f"   Seeds: {SEEDS}")
    print(f"   Samples per type: {SAMPLES_PER_TYPE}")
    print(f"   This will create {len(SEEDS)} directories:")
    for seed in SEEDS:
        print(f"      - {base_destination_folder}/seed_{seed}")

    # Create the datasets
    try:
        all_results = extract_multiple_seed_datasets(
            source_folder=source_folder,
            base_destination_folder=base_destination_folder,
            flower_types=flower_types,
            seeds=SEEDS,
            samples_per_type=SAMPLES_PER_TYPE
        )

        print(f"\n🎉 All datasets created successfully!")

        # Verify the datasets
        verify_all_datasets(base_destination_folder)

        print(f"\n📁 Your datasets are ready at:")
        for seed in SEEDS:
            dataset_path = Path(base_destination_folder) / f"seed_{seed}"
            print(f"   {dataset_path}")

    except Exception as e:
        print(f"❌ Error during dataset creation: {e}")