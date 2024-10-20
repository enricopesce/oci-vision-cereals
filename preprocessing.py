import os
import json
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from pathlib import Path
import random

# Set the path for the dataset and output path for processed images
input_dir = "original/"
output_dir = "processed/"
os.makedirs(output_dir, exist_ok=True)

# Path for the JSONL metadata file
jsonl_path = output_dir + "/metadata.jsonl"

# Define target size for resizing
TARGET_SIZE = (224, 224)

# Augmentation parameters
AUGMENTATION_PER_IMAGE = 3  # Number of augmented versions per original image
ROTATION_RANGE = (-20, 20)  # Degrees
BRIGHTNESS_RANGE = (0.8, 1.2)
CONTRAST_RANGE = (0.8, 1.2)
FLIP_PROBABILITY = 0.5

DATALABEL_DATA = {
    "displayName": "grains",
    "labelsSet": [{"name": "sorghum"}, {"name": "corn"}, {"name": "wheat"}],
    "annotationFormat": "MULTI_LABEL",
    "datasetFormatDetails": {"formatType": "IMAGE"},
}

def apply_augmentations(image):
    """Apply random augmentations to the image"""
    augmented_images = []
    
    for _ in range(AUGMENTATION_PER_IMAGE):
        img_aug = image.copy()
        
        # Random rotation
        if random.random() < 0.7:
            rotation_angle = random.uniform(*ROTATION_RANGE)
            img_aug = img_aug.rotate(rotation_angle, Image.Resampling.BILINEAR, expand=True)
        
        # Random brightness adjustment
        if random.random() < 0.7:
            brightness_factor = random.uniform(*BRIGHTNESS_RANGE)
            enhancer = ImageEnhance.Brightness(img_aug)
            img_aug = enhancer.enhance(brightness_factor)
        
        # Random contrast adjustment
        if random.random() < 0.7:
            contrast_factor = random.uniform(*CONTRAST_RANGE)
            enhancer = ImageEnhance.Contrast(img_aug)
            img_aug = enhancer.enhance(contrast_factor)
        
        # Random horizontal flip
        if random.random() < FLIP_PROBABILITY:
            img_aug = ImageOps.mirror(img_aug)
        
        # Random color jitter
        if random.random() < 0.5:
            color_factor = random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Color(img_aug)
            img_aug = enhancer.enhance(color_factor)
            
        # Random sharpness
        if random.random() < 0.3:
            sharpness_factor = random.uniform(0.8, 1.5)
            enhancer = ImageEnhance.Sharpness(img_aug)
            img_aug = enhancer.enhance(sharpness_factor)

        augmented_images.append(img_aug)
    
    return augmented_images

def resize_and_crop(image, target_size):
    target_width, target_height = target_size
    original_width, original_height = image.size

    # Calculate aspect ratios
    aspect_ratio_original = original_width / original_height
    aspect_ratio_target = target_width / target_height

    # Resize based on the aspect ratio
    if aspect_ratio_original > aspect_ratio_target:
        new_height = target_height
        new_width = int(aspect_ratio_original * target_height)
    else:
        new_width = target_width
        new_height = int(target_width / aspect_ratio_original)

    # Resize the image
    img_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Calculate the cropping area
    left = (new_width - target_width) / 2
    top = (new_height - target_height) / 2
    right = (new_width + target_width) / 2
    bottom = (new_height + target_height) / 2

    # Crop the center
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
    try:
        # Open and preprocess original image
        img = Image.open(image_path)
        
        # Generate augmented versions
        augmented_images = apply_augmentations(img)
        all_images = [img] + augmented_images  # Original + augmented
        
        for idx, img_variant in enumerate(all_images):
            # Apply standard preprocessing
            img_resized = resize_and_crop(img_variant, target_size)
            img_rgb = colorspace_image(img_resized)
            img_final = normalize_image(img_rgb)
            
            # Generate unique filename for each variant
            variant_id = f"{image_id}_{idx}" if idx > 0 else str(image_id)
            output_filename = f"{variant_id}.jpg"
            output_path = os.path.join(output_dir, label)
            output_full_path = os.path.join(output_dir, label, output_filename)
            os.makedirs(output_path, exist_ok=True)

            p = Path(output_full_path)
            relative_output_path = str(Path(*p.parts[1:]))
            
            # Save image
            img_final.save(output_full_path, "JPEG", quality=95)

            # Create and save metadata
            metadata = {
                "sourceDetails": {"path": relative_output_path},
                "annotations": [
                    {"entities": [{"entityType": "GENERIC", "labels": [{"label_name": label}]}]}
                ],
                "augmentation": "original" if idx == 0 else f"augmented_{idx}"
            }
            jsonl_file.write(json.dumps(metadata, separators=(",", ":")) + "\n")
            
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

# Main processing loop
image_id = 1
with open(jsonl_path, "w") as jsonl_file:
    jsonl_file.write(json.dumps(DATALABEL_DATA, separators=(",", ":")) + "\n")
    for subdir, _, files in os.walk(input_dir):
        label = os.path.basename(subdir)
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(subdir, file)
                output_path = os.path.join(output_dir, f"{image_id}.jpg")
                print(f"Processing {file_path}...")
                preprocess_image(file_path, output_path, TARGET_SIZE, image_id, label, jsonl_file)
                image_id += 1

print(f"OCI Data Labeling JSONL metadata saved to {jsonl_path}")

https://objectstorage.eu-frankfurt-1.oraclecloud.com/n/frddomvd8z4q/b/vision/o/metadata.jsonl
