import os
import re
import json
import math
import random
import logging
from PIL import Image, ImageEnhance
import pandas as pd
import time

# Configuration
AUGMENTATIONS = [
    ('rotate', {'angle': 90}),
    ('flip', {'direction': 'horizontal'}),
    ('shear', {'factor': 0.2}),
    ('brightness', {'factor': 0.8}),
]


TARGET_FRACTURED_COUNT = 3000
IMAGES_DIR = 'images/Fractured'
CSV_PATH = 'dataset.csv'
CSV_AUG_PATH = 'dataset_augmented.csv'
JSON_PATH = 'Annotations/COCO JSON/COCO_fracture_masks.json'
JSON_AUG_PATH = 'Annotations/COCO JSON/COCO_fracture_masks_augmented.json'
LOG_PATH = 'augmentation_logs.log'

# Setup logging
logging.basicConfig(
    filename=LOG_PATH,
    filemode='a',
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)

def get_next_img_names(img_dir, n_needed):
    """Find the next available IMG000XXXX.jpg file names in img_dir."""
    existing = set()
    pattern = re.compile(r'IMG(\d{7})\.jpg')
    for fname in os.listdir(img_dir):
        match = pattern.fullmatch(fname)
        if match:
            existing.add(int(match.group(1)))
    next_names = []
    candidate = 1
    while len(next_names) < n_needed:
        if candidate not in existing:
            next_names.append(f"IMG{candidate:07d}.jpg")
        candidate += 1
    return next_names

def transform_point(x, y, img_width, img_height, transform):
    if transform['type'] == 'rotate':
        angle = math.radians(transform['angle'])
        cx, cy = img_width/2, img_height/2
        x_new = cx + math.cos(angle)*(x - cx) - math.sin(angle)*(y - cy)
        y_new = cy + math.sin(angle)*(x - cx) + math.cos(angle)*(y - cy)
        return x_new, y_new
    elif transform['type'] == 'flip':
        if transform['direction'] == 'horizontal':
            return img_width - x, y
        # TODO: Uncomment this when we have vertical flips
        # return x, img_height - y
    elif transform['type'] == 'shear':
        shear = transform['factor']
        return x + shear*y, y
    return x, y  # For brightness changes

def augment_image(img, transform):
    if transform['type'] == 'rotate':
        return img.rotate(transform['angle'], expand=True)
    elif transform['type'] == 'flip':
        return img.transpose(Image.FLIP_LEFT_RIGHT if transform['direction'] == 'horizontal' else Image.FLIP_TOP_BOTTOM)
    elif transform['type'] == 'shear':
        shear = transform['factor']
        return img.transform(img.size, Image.AFFINE, (1, shear, 0, 0, 1, 0))
    elif transform['type'] == 'brightness':
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(transform['factor'])
    return img

# Load existing data
df = pd.read_csv(CSV_PATH)
if os.path.exists(CSV_AUG_PATH):
    os.remove(CSV_AUG_PATH)
df.to_csv(CSV_AUG_PATH, index=False)  # Start with original CSV

with open(JSON_PATH, 'r') as f:
    coco_data = json.load(f)
if os.path.exists(JSON_AUG_PATH):
    os.remove(JSON_AUG_PATH)
with open(JSON_AUG_PATH, 'w') as f:
    json.dump(coco_data, f)

# Filter fractured images
fractured_df = df[df['fractured'] == 1]
current_fractured = len(fractured_df)
augmentations_needed = TARGET_FRACTURED_COUNT - current_fractured

print(f"Original fractured images: {current_fractured}")
print(f"Augmentations needed: {augmentations_needed}")

# Get current max IDs
max_image_id = max(img['id'] for img in coco_data['images'])
max_ann_id = max(ann['id'] for ann in coco_data['annotations'])

# Get next available image names
next_names = get_next_img_names(IMAGES_DIR, augmentations_needed)
next_name_idx = 0
augmented_count = 0

start_time = time.time()

for _, row in fractured_df.iterrows():
    if augmented_count >= augmentations_needed:
        break

    with open(JSON_AUG_PATH, 'r') as f:
        coco_data_aug = json.load(f)

    original_img = next((img for img in coco_data_aug['images'] if img['file_name'] == row['image_id']), None)
    if not original_img:
        continue

    original_anns = [ann for ann in coco_data_aug['annotations'] if ann['image_id'] == original_img['id']]

    for aug_type, params in AUGMENTATIONS:
        if augmented_count >= augmentations_needed:
            break

        img_path = os.path.join(IMAGES_DIR, row['image_id'])
        with Image.open(img_path) as img:
            transformed_img = augment_image(img, {'type': aug_type, **params})
            new_filename = next_names[next_name_idx]
            next_name_idx += 1
            transformed_img.save(os.path.join(IMAGES_DIR, new_filename))

        # Update CSV immediately
        new_row = row.copy()
        new_row['image_id'] = new_filename
        pd.DataFrame([new_row]).to_csv(CSV_AUG_PATH, mode='a', header=False, index=False)

        # Update COCO images immediately
        max_image_id += 1
        new_image_entry = {
            'id': max_image_id,
            'width': transformed_img.width,
            'height': transformed_img.height,
            'file_name': new_filename
        }
        coco_data_aug['images'].append(new_image_entry)

        # Update COCO annotations immediately
        new_ann_list = []
        for ann in original_anns:
            max_ann_id += 1
            new_ann = ann.copy()
            new_ann['id'] = max_ann_id
            new_ann['image_id'] = max_image_id

            transformed_seg = []
            for i in range(0, len(ann['segmentation'][0]), 2):
                x, y = ann['segmentation'][0][i], ann['segmentation'][0][i+1]
                x_new, y_new = transform_point(
                    x, y,
                    original_img['width'], original_img['height'],
                    {'type': aug_type, **params}
                )
                transformed_seg.extend([x_new, y_new])

            new_ann['segmentation'] = [transformed_seg]
            xs = transformed_seg[::2]
            ys = transformed_seg[1::2]
            new_ann['bbox'] = [
                min(xs), min(ys),
                max(xs) - min(xs),
                max(ys) - min(ys)
            ]
            coco_data_aug['annotations'].append(new_ann)
            new_ann_list.append(new_ann)

        # Write updated JSON immediately
        with open(JSON_AUG_PATH, 'w') as f:
            json.dump(coco_data_aug, f)

        # Log everything
        logging.info(f"Augmentation: {aug_type} {params}")
        print(f"Augmentation: {aug_type} {params}")
        
        logging.info(f"Input: {img_path}")
        print(f"Input: {img_path}")
        
        logging.info(f"Output: {os.path.join(IMAGES_DIR, new_filename)}")
        print(f"Output: {os.path.join(IMAGES_DIR, new_filename)}")
        
        # Log input (original) image entry from JSON
        logging.info(f"Input image entry for JSON: {json.dumps(original_img)}")        
        logging.info(f"New image entry for JSON: {json.dumps(new_image_entry)}")
        # print(f"New image entry for JSON: {json.dumps(new_image_entry)}")
        
        # Log input (original) annotation(s) from JSON
        for ann in original_anns:
            logging.info(f"Input annotation entry for JSON: {json.dumps(ann)}")        
        # Log new annotation(s) from JSON
        for ann in new_ann_list:
            logging.info(f"New annotation entry for JSON: {json.dumps(ann)}")
            # print(f"New annotation entry for JSON: {json.dumps(ann)}")
        
        # Log input (original) CSV row
        logging.info(f"Input CSV row: {row.to_dict()}")
        logging.info(f"New CSV row: {new_row.to_dict()}")
        # print(f"New CSV row: {new_row.to_dict()}")
        
        logging.info(f"Total augmented: {augmented_count+1}/{augmentations_needed}")
        print(f"Total augmented: {augmented_count+1}/{augmentations_needed}")
        logging.info("\n\n=================================\n\n")
        # print("\n\n=================================\n\n")

        print(f"Augmented and saved {new_filename}, updated CSV and JSON, and logged all info.")
        augmented_count += 1
        # input("Press Enter to continue...")

print(f"Time taken: {time.time() - start_time} seconds")
print("\nAugmentation complete. Check the log file (augmentation_log.txt) for all details.")
print("You can now randomly verify records in the CSV and JSON files.")
