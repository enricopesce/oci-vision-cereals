import os
import json
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random

# Core configuration
INPUT_DIR = "original/"
OUTPUT_DIR = "processed/"
TARGET_SIZE = (224, 224)
AUGMENTATIONS_PER_IMAGE = 3

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_augmentations(image):
    """Apply basic image augmentations."""
    augmented_images = []
    
    for _ in range(AUGMENTATIONS_PER_IMAGE):
        try:
            img_aug = image.copy()
            
            # Apply random rotation
            if random.random() < 0.7:
                angle = random.uniform(-20, 20)
                img_aug = img_aug.rotate(angle, Image.Resampling.BILINEAR, expand=True, fillcolor=255)
            
            # Apply random brightness
            if random.random() < 0.7:
                factor = random.uniform(0.8, 1.2)
                img_aug = ImageEnhance.Brightness(img_aug).enhance(factor)
            
            # Apply random horizontal flip
            if random.random() < 0.5:
                img_aug = ImageOps.mirror(img_aug)
            
            augmented_images.append(img_aug)
            
        except Exception as e:
            print(f"Augmentation failed: {str(e)}")
            continue
            
    return augmented_images

def resize_and_crop(image, target_size):
    """Resize and crop image to target size maintaining aspect ratio."""
    target_width, target_height = target_size
    original_width, original_height = image.size
    
    # Calculate aspect ratios
    aspect_ratio_original = original_width / original_height
    aspect_ratio_target = target_width / target_height
    
    # Resize based on aspect ratio
    if aspect_ratio_original > aspect_ratio_target:
        new_height = target_height
        new_width = int(aspect_ratio_original * target_height)
    else:
        new_width = target_width
        new_height = int(target_width / aspect_ratio_original)
    
    # Resize and crop
    img_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    return img_resized.crop((left, top, left + target_width, top + target_height))

def process_image(image_path, output_path, image_id, label):
    """Process a single image and its augmentations."""
    try:
        # Load and process original image
        img = Image.open(image_path).convert('RGB')
        img = resize_and_crop(img, TARGET_SIZE)
        
        # Process original and augmented versions
        all_images = [img] + apply_augmentations(img)
        
        # Save images and generate metadata
        metadata_entries = []
        for idx, img_variant in enumerate(all_images):
            # Save image
            variant_id = f"{image_id}_{idx}" if idx > 0 else str(image_id)
            filename = f"{variant_id}.jpg"
            output_subdir = os.path.join(OUTPUT_DIR, label)
            os.makedirs(output_subdir, exist_ok=True)
            
            output_path = os.path.join(output_subdir, filename)
            img_variant.save(output_path, "JPEG", quality=95)
            
            # Generate metadata
            metadata = {
                "sourceDetails": {"path": f"{label}/{filename}"},
                "annotations": [{
                    "entities": [{
                        "entityType": "GENERIC",
                        "labels": [{"label_name": label}]
                    }]
                }]
            }
            metadata_entries.append(metadata)
            
        return metadata_entries
            
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return []

def main():
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.jsonl")
    image_id = 1
    
    with open(metadata_path, 'w') as jsonl_file:
        # Process all images
        for subdir, _, files in os.walk(INPUT_DIR):
            label = os.path.basename(subdir)
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(subdir, file)
                    print(f"Processing {file_path}...")
                    
                    metadata_entries = process_image(file_path, OUTPUT_DIR, image_id, label)
                    for entry in metadata_entries:
                        jsonl_file.write(json.dumps(entry) + '\n')
                    
                    image_id += 1
    
    print(f"Processing complete. Metadata saved to {metadata_path}")

if __name__ == "__main__":
    main()