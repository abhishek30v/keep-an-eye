import os
import shutil
from glob import glob
from PIL import Image       # for getting image dimensions that is used for normalization.

# 👇 Class name to YOLO class ID mapping
animals_map = {
    'Tiger' : 0,
    'Leopard' : 1, 
    'Cheetah' : 2, 
    'Elephant' : 3, 
    'Monkey' : 4, 
    'Deer' : 5, 
    'Lion' : 6, 
    'Bear' : 7, 
    'Pig' : 8, 
    'Bull' : 9
}

def create_dirs(base):
    """Create YOLOv8-compliant folder structure"""
    for split in ['train', 'test']:
        os.makedirs(f"{base}/images/{split}", exist_ok=True)
        os.makedirs(f"{base}/labels/{split}", exist_ok=True)

def convert_labels_for_class(animal, split, output_base):
    """
    Convert OID labels (with Label subfolder) to YOLO format
    and copy images/labels to YOLOv8 structure
    """
    input_dir = f"OIDv4_ToolKit/OID/Dataset/{split}/{animal}"
    label_dir = os.path.join(input_dir, "Label")
    print(f"\n🔍 Processing class: {animal} [{split}]")
    print(f"📂 Checking for label files in: {label_dir}")

    label_files = glob(os.path.join(label_dir, "*.txt"))
    print(f"📁 Found {len(label_files)} label files")

    for label_path in label_files:
        image_filename = os.path.basename(label_path).replace(".txt", ".jpg")
        image_path = os.path.join(input_dir, image_filename)

        if not os.path.exists(image_path):
            print(f"⚠️ Image missing for label: {label_path}")
            continue

        img = Image.open(image_path)
        img_width, img_height = img.size

        with open(label_path, "r") as f:
            lines = f.readlines()

        yolo_labels = ""
        for line in lines:
            print(f"🔎 Raw label line: {line.strip()}")
            parts = line.strip().split()
            if len(parts) != 5 or parts[0] not in animals_map:
                print("⛔ Skipping invalid or unknown class line")
                continue

            cls_id = animals_map[parts[0]]
            xmin, xmax = float(parts[1]), float(parts[3])
            ymin, ymax = float(parts[2]), float(parts[4])
            
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            height = (ymax - ymin) / img_height
            width = (xmax - xmin) / img_width


            yolo_labels += f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

        # Save image and label
        split_folder = 'train' if split == 'train' else 'test'
        dest_img = os.path.join(output_base, 'images', split_folder, image_filename)
        dest_lbl = os.path.join(output_base, 'labels', split_folder, image_filename.replace('.jpg', '.txt'))

        shutil.copy(image_path, dest_img)
        print(f"✅ Copied image → {dest_img}") # for debugging
        with open(dest_lbl, "w") as f:
            f.write(yolo_labels)
        print(f"✅ Wrote labels → {dest_lbl}") # for debugging

# Define the animals(or classes) and run
animals = ['Tiger', 'Leopard', 'Cheetah', 'Elephant', 'Monkey', 'Deer', 'Lion', 'Bear', 'Pig', 'Bull']  # Add all your classes here
OUTPUT_BASE = "Animals"

create_dirs(OUTPUT_BASE)
for animal in animals:
    convert_labels_for_class(animal, 'train', OUTPUT_BASE)
    convert_labels_for_class(animal, 'test', OUTPUT_BASE)
