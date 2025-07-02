import os
import subprocess


def download_class(class_name, num_images, dataset_type):
    """
    This is the function used to download images of particular class from Open Images using OIDv4 Toolkit.

    Parameters:
        class_name (string)-> The class to download (ex-> Elephant, Tiger, etc.).
        num_images (integer)-> Number of images to download.
        dataset_type (string)-> 'train' or 'test'.
    """
    dataset_type = dataset_type.lower()
    if dataset_type not in ['train', 'test']:
        raise ValueError("dataset_type must be 'train' or 'test'")

    toolkit_path = "OIDv4_ToolKit"
    os.makedirs(toolkit_path, exist_ok=True)

    cmd = [                         # This is the command that would run in terminal, using subprocess.run();
        "python", "main.py", "downloader",
        "--classes", class_name,
        "--type_csv", dataset_type,
        "--limit", str(num_images)
    ]

    print(f"\n🚀 Downloading {num_images} images for class '{class_name}' as {dataset_type} set...\n")
    subprocess.run(cmd, cwd=toolkit_path)


# Modify this list and counts as needed
animals = ['Tiger', 'Leopard', 'Cheetah', 'Elephant', 'Monkey', 'Deer', 'Lion', 'Bear', 'Pig', 'Bull']
train_count = 600
test_count = 150

for animal in animals:
    download_class(animal, train_count, 'train')
    download_class(animal, test_count, 'test')
