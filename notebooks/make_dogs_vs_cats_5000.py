from pathlib import Path
import random
import shutil

# ============================================================
# FIXED BASE PATHS (use the TRAIN folder with labeled images)
# ============================================================

# Project root = one folder above /notebooks
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Use the 'train' subfolder where cat.XXXX.jpg and dog.XXXX.jpg live
FULL_DIR = PROJECT_ROOT / "data" / "dogs_vs_cats_full" / "train"
OUT_ROOT = PROJECT_ROOT / "data" / "dogs_vs_cats_5000"

print("Project root:", PROJECT_ROOT)
print("Full dataset:", FULL_DIR)
print("Output root: ", OUT_ROOT)

# ============================================================
# CONFIGURATION
# ============================================================

N_PER_CLASS = 2500  # 2500 cats + 2500 dogs = 5000

SPLIT_COUNTS = {
    "train": 1500,       # 1500 cats + 1500 dogs = 3000
    "validation": 500,   # 500 cats + 500 dogs = 1000
    "test": 500,         # 500 cats + 500 dogs = 1000
}

CLASS_NAMES = ["cat", "dog"]

random.seed(42)  # reproducibility


# ============================================================
# HELPERS
# ============================================================

def sample_files_for_class(class_name: str):
    """Randomly select N_PER_CLASS images for a given class."""
    pattern = f"{class_name}.*.jpg"
    files = sorted(FULL_DIR.glob(pattern))

    if len(files) < N_PER_CLASS:
        raise ValueError(
            f"Not enough {class_name} images in {FULL_DIR}. "
            f"Found {len(files)}, need {N_PER_CLASS}."
        )

    return random.sample(files, N_PER_CLASS)


# ============================================================
# MAIN SCRIPT
# ============================================================

def main():
    print("\n=== Starting dataset creation ===\n")

    if not FULL_DIR.exists():
        raise FileNotFoundError(f"Full dataset folder not found: {FULL_DIR}")

    # Create output folder structure
    for split in SPLIT_COUNTS.keys():
        for cls in CLASS_NAMES:
            out_dir = OUT_ROOT / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)

    # Sample the images
    subsets = {}
    for cls in CLASS_NAMES:
        print(f"Sampling {N_PER_CLASS} images for class '{cls}'...")
        subsets[cls] = sample_files_for_class(cls)
        print(f"✓ Sampled {len(subsets[cls])} images for '{cls}'")

    # Copy images into train/val/test
    for cls in CLASS_NAMES:
        files = subsets[cls]
        start = 0

        for split, count in SPLIT_COUNTS.items():
            end = start + count
            split_files = files[start:end]

            out_dir = OUT_ROOT / split / cls
            print(f"\nCopying {len(split_files)} '{cls}' images → {split}/...")

            for src in split_files:
                dst = out_dir / src.name
                shutil.copy2(src, dst)

            start = end

    # Summary
    print("\n=== Summary ===")
    for split in SPLIT_COUNTS.keys():
        for cls in CLASS_NAMES:
            n = len(list((OUT_ROOT / split / cls).glob("*.jpg")))
            print(f"{split}/{cls}: {n} images")

    print("\n✓ Done! 5000-image subset created at:")
    print(OUT_ROOT)


if __name__ == "__main__":
    main()
