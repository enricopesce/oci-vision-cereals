import os
import json
from PIL import Image
import numpy as np
from pathlib import Path

# Set the path for the dataset and output path for processed images
input_dir = "original/"
output_dir = "processed/"
os.makedirs(output_dir, exist_ok=True)

# Path for the JSONL metadata file
jsonl_path = output_dir + "/metadata.jsonl"

# Define target size for resizing
TARGET_SIZE = (224, 224)

DATALABEL_DATA = {
    "displayName": "grains",
    "labelsSet": [{"name": "sorghum"}, {"name": "corn"}, {"name": "wheat"}],
    "annotationFormat": "MULTI_LABEL",
    "datasetFormatDetails": {"formatType": "IMAGE"},
}


def resize_and_crop(image, target_size):
    target_width, target_height = target_size
    original_width, original_height = image.size

    # Calculate aspect ratios
    aspect_ratio_original = original_width / original_height
    aspect_ratio_target = target_width / target_height

    # Resize based on the aspect ratio
    if aspect_ratio_original > aspect_ratio_target:
        # Image is wider, resize by height and crop width
        new_height = target_height
        new_width = int(aspect_ratio_original * target_height)
    else:
        # Image is taller or equal in aspect, resize by width and crop height
        new_width = target_width
        new_height = int(target_width / aspect_ratio_original)

    # Resize the image (this will make one dimension match, the other larger)
    img_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Calculate the cropping area to make the final image exactly the target size
    left = (new_width - target_width) / 2
    top = (new_height - target_height) / 2
    right = (new_width + target_width) / 2
    bottom = (new_height + target_height) / 2

    # Crop the center of the resized image
    img_cropped = img_resized.crop((left, top, right, bottom))

    return img_cropped


def normalize_image(image):
    np_image = np.array(image)
    normalized_image = np_image / 255.0
    np_image_255 = (normalized_image * 255).astype(np.uint8)
    return Image.fromarray(np_image_255)


def colorspace_image(image):
    return image.convert("RGB")


def preprocess_image(image_path, output_path, target_size, image_id, label, jsonl_file):
    # Open the image
    img = Image.open(image_path)

    img_resized = resize_and_crop(img, target_size)
    img_rgb = colorspace_image(img_resized)
    img_final = normalize_image(img_rgb)
    
    # Save the preprocessed image with progressive ID as the filename (e.g., "1.jpg", "2.jpg", etc.)
    output_filename = f"{image_id}.jpg"
    output_path = os.path.join(output_dir, label)
    output_full_path = os.path.join(output_dir, label, output_filename)
    os.makedirs(output_path, exist_ok=True)

    p = Path(output_full_path)
    output_path = str(Path(*p.parts[1:]))
    img_final.save(output_full_path, "JPEG")

    metadata = {
        "sourceDetails": {"path": output_path},
        "annotations": [
            {"entities": [{"entityType": "GENERIC", "labels": [{"label_name": label}]}]}
        ],
    }

    # Write metadata to JSONL file (each line is a separate JSON object)
    jsonl_file.write(json.dumps(metadata, separators=(",", ":")) + "\n")


# Loop through images and preprocess only .jpg files
image_id = 1
with open(jsonl_path, "w") as jsonl_file:  # Open JSONL file for writing
    jsonl_file.write(json.dumps(DATALABEL_DATA, separators=(",", ":")) + "\n")
    for subdir, _, files in os.walk(input_dir):
        label = os.path.basename(
            subdir
        )  # Assuming subdirectory name is the label/class
        for file in files:
            if file.lower().endswith(".jpg"):  # Only process .jpg files
                file_path = os.path.join(subdir, file)
                output_path = os.path.join(output_dir, f"{image_id}.jpg")
                preprocess_image(
                    file_path, output_path, TARGET_SIZE, image_id, label, jsonl_file
                )
                image_id += 1

print(f"OCI Data Labeling JSONL metadata saved to {jsonl_path}")


# https://objectstorage.eu-frankfurt-1.oraclecloud.com/n/frddomvd8z4q/b/vision/o/data.json