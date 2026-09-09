from pathlib import Path
import shutil

# Dataset locations
DATASET_ROOT = Path.home() / "thesis/datasets/VisDrone"

SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
}

# VisDrone category -> YOLO class ID
CATEGORY_MAP = {
    1: 0,   # pedestrian
    2: 1,   # people
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    7: 6,   # tricycle
    8: 7,   # awning-tricycle
    9: 8,   # bus
    10: 9,  # motor
}


def convert_split(split_name, folder_name):
    src = DATASET_ROOT / folder_name

    image_dir = src / "images"
    annotation_dir = src / "annotations"

    output_image_dir = DATASET_ROOT / "images" / split_name
    output_label_dir = DATASET_ROOT / "labels" / split_name

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    annotation_files = sorted(annotation_dir.glob("*.txt"))

    print(f"\nProcessing {split_name}...")
    print(f"Annotations found: {len(annotation_files)}")

    converted = 0
    skipped = 0

    for ann_file in annotation_files:

        # Find corresponding image
        image_file = image_dir / f"{ann_file.stem}.jpg"

        if not image_file.exists():
            print(f"WARNING: Image not found: {image_file}")
            skipped += 1
            continue

        # Read image dimensions
        try:
            from PIL import Image
            with Image.open(image_file) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"WARNING: Cannot read {image_file}: {e}")
            skipped += 1
            continue

        yolo_lines = []

        with open(ann_file, "r") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                values = line.split(",")

                if len(values) != 8:
                    print(f"WARNING: Invalid annotation: {ann_file}")
                    continue

                try:
                    x, y, w, h, score, category, truncation, occlusion = map(
                        int, values
                    )
                except ValueError:
                    print(f"WARNING: Invalid values in {ann_file}")
                    continue

                # Ignore invalid / ignored regions
                if category not in CATEGORY_MAP:
                    continue

                if w <= 0 or h <= 0:
                    continue

                # Convert VisDrone bbox -> YOLO normalized bbox
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height

                class_id = CATEGORY_MAP[category]

                # Clamp values to valid YOLO range
                x_center = min(max(x_center, 0.0), 1.0)
                y_center = min(max(y_center, 0.0), 1.0)
                width = min(max(width, 0.0), 1.0)
                height = min(max(height, 0.0), 1.0)

                yolo_lines.append(
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}\n"
                )

        # Write YOLO label
        output_label_file = output_label_dir / f"{ann_file.stem}.txt"

        with open(output_label_file, "w") as f:
            f.writelines(yolo_lines)

        # Copy image
        shutil.copy2(image_file, output_image_dir / image_file.name)

        converted += 1

    print(f"Converted: {converted}")
    print(f"Skipped:   {skipped}")


if __name__ == "__main__":
    for split, folder in SPLITS.items():
        convert_split(split, folder)

    print("\nConversion completed!")
