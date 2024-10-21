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


def apply_augmentations(image, config=None):
    """
    Apply configurable augmentations to the image with improved error handling and efficiency.

    Args:
        image: PIL Image object
        config: Optional dictionary with augmentation parameters
    Returns:
        list of augmented PIL Image objects
    """
    if config is None:
        config = {
            "num_augmentations": 5,
            "rotation": {"prob": 0.7, "range": (-20, 20)},
            "brightness": {"prob": 0.7, "range": (0.8, 1.2)},
            "contrast": {"prob": 0.7, "range": (0.8, 1.2)},
            "flip": {"prob": 0.5},
            "color": {"prob": 0.5, "range": (0.8, 1.2)},
            "sharpness": {"prob": 0.3, "range": (0.8, 1.5)},
            "blur": {"prob": 0.2, "range": (0.5, 1.5)},
            "noise": {"prob": 0.2, "intensity": (5, 20)},
        }

    augmented_images = []

    for _ in range(config["num_augmentations"]):
        try:
            img_aug = image.copy()

            # Rotation with boundary handling
            if random.random() < config["rotation"]["prob"]:
                rotation_angle = random.uniform(*config["rotation"]["range"])
                # Use expand=True to prevent image cropping and fill=255 for white background
                img_aug = img_aug.rotate(
                    rotation_angle,
                    Image.Resampling.BILINEAR,
                    expand=True,
                    fillcolor=255,
                )

            # Brightness adjustment with error handling
            if random.random() < config["brightness"]["prob"]:
                brightness_factor = random.uniform(*config["brightness"]["range"])
                enhancer = ImageEnhance.Brightness(img_aug)
                img_aug = enhancer.enhance(brightness_factor)

            # Contrast adjustment
            if random.random() < config["contrast"]["prob"]:
                contrast_factor = random.uniform(*config["contrast"]["range"])
                enhancer = ImageEnhance.Contrast(img_aug)
                img_aug = enhancer.enhance(contrast_factor)

            # Horizontal flip
            if random.random() < config["flip"]["prob"]:
                img_aug = ImageOps.mirror(img_aug)

            # Color adjustment
            if random.random() < config["color"]["prob"]:
                color_factor = random.uniform(*config["color"]["range"])
                enhancer = ImageEnhance.Color(img_aug)
                img_aug = enhancer.enhance(color_factor)

            # Sharpness adjustment
            if random.random() < config["sharpness"]["prob"]:
                sharpness_factor = random.uniform(*config["sharpness"]["range"])
                enhancer = ImageEnhance.Sharpness(img_aug)
                img_aug = enhancer.enhance(sharpness_factor)

            # Gaussian blur (new)
            if random.random() < config["blur"]["prob"]:
                from PIL import ImageFilter

                blur_radius = random.uniform(*config["blur"]["range"])
                img_aug = img_aug.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            # Random noise (new)
            if random.random() < config["noise"]["prob"]:
                img_array = np.array(img_aug)
                noise = np.random.normal(
                    0, random.uniform(*config["noise"]["intensity"]), img_array.shape
                )
                noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
                img_aug = Image.fromarray(noisy_array)

            augmented_images.append(img_aug)

        except Exception as e:
            print(f"Warning: Augmentation failed with error: {str(e)}")
            continue

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


def normalize_minmax(image: Image.Image, target_range: tuple = (0, 1)) -> np.ndarray:
    """
    Normalize PIL Image using min-max scaling.

    Args:
        image: Input PIL Image
        target_range: Tuple of (min, max) for target range, default (0,1)

    Returns:
        Normalized image as numpy array with values in target range
    """
    # Convert PIL Image to numpy array
    np_image = np.array(image).astype(np.float32)

    # Initialize normalized array
    normalized = np.zeros_like(np_image, dtype=np.float32)

    # Normalize each channel independently
    for channel in range(3):  # Assuming RGB image
        channel_data = np_image[:, :, channel]
        min_val = channel_data.min()
        max_val = channel_data.max()

        if max_val - min_val != 0:
            normalized[:, :, channel] = (channel_data - min_val) / (max_val - min_val)
        else:
            normalized[:, :, channel] = 0  # Handle constant-value channel

    # Scale to target range
    min_target, max_target = target_range
    normalized = normalized * (max_target - min_target) + min_target

    return normalized


def denormalize_minmax(normalized_image: np.ndarray) -> Image.Image:
    """
    Convert normalized image back to PIL Image format.

    Args:
        normalized_image: Normalized image array

    Returns:
        PIL Image with values in [0, 255] range
    """
    # Scale back to [0, 255] range
    denormalized = normalized_image * 255.0

    # Clip values and convert to uint8
    denormalized = np.clip(denormalized, 0, 255).astype(np.uint8)

    return Image.fromarray(denormalized)


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
            img_final = denormalize_minmax(normalize_minmax(img_rgb))

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
                    {
                        "entities": [
                            {"entityType": "GENERIC", "labels": [{"label_name": label}]}
                        ]
                    }
                ],
                "augmentation": "original" if idx == 0 else f"augmented_{idx}",
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
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                file_path = os.path.join(subdir, file)
                output_path = os.path.join(output_dir, f"{image_id}.jpg")
                print(f"Processing {file_path}...")
                preprocess_image(
                    file_path, output_path, TARGET_SIZE, image_id, label, jsonl_file
                )
                image_id += 1

print(f"OCI Data Labeling JSONL metadata saved to {jsonl_path}")

# https://objectstorage.eu-frankfurt-1.oraclecloud.com/n/frddomvd8z4q/b/vision/o/metadata.jsonl
