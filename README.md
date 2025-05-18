# Bone Fracture Dataset Preprocessing Scripts

## Overview
This repository contains scripts used to preprocess the [FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation of Musculoskeletal Radiographs](https://figshare.com/articles/dataset/The_dataset/22363012?file=43283628) for the research paper "ResNet50-Driven Bone Fracture Detection with Attention-Augmented U-Net Segmentation" by Muhammad Ilyas. The scripts address corrupted images and class imbalance in the original dataset, which is licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Repository Contents
- **`remove_corrupted_images.py`**: Removes 59 corrupted images from the non-fractured class (originally 3,366 images, reduced to 3,307) and updates `dataset.csv`.
- **`augment_data.py`**: Augments fractured images from 717 to 3,000 using 90° rotation, horizontal flip, 0.2 shear, and 0.8 brightness adjustment, updating `dataset.csv'and `COCO_fracture_masks.json`.
- **`corrupted_images.txt`**: Lists the 59 corrupted image filenames removed from the non-fractured class.

## Purpose
The scripts improve the FracAtlas dataset by:
- Removing 59 corrupted non-fractured images that caused training errors.
- Augmenting fractured images to 3,000 to address class imbalance (originally 717 fractured vs. 3,366 non-fractured images).
These changes enhance data integrity and model performance for fracture detection and segmentation.

## Usage
- Download the [FracAtlas dataset](https://figshare.com/articles/dataset/The_dataset/22363012?file=43283628).
- Use `remove_corrupted_images.py` to delete corrupted images and update `dataset.csv`.
- Run `augment_data.py` to augment fractured images and update `dataset.csv` and `COCO_fracture_masks.json`.
- Note: Medical and radiology expertise is recommended, as per the original dataset.

## Credits
- **Original Dataset**: FracAtlas by [Iftekharul Abedeen](https://figshare.com/authors/Iftekharul_Abedeen/14603630), Md. Ashiqur Rahman, Fatema Zohra Prottyasha, Tasnim Ahmed, Tareque Mohmud Chowdhury, and Swakkhar Shatabda.
- **Updated Dataset**: Muhammad Ilyas, for the research paper "ResNet50-Driven Bone Fracture Detection with Attention-Augmented U-Net Segmentation"

## Citation
If using these scripts or the processed dataset, please cite:
- Original Dataset:
  ```
  Abedeen, I., Rahman, M. A., Prottyasha, F. Z., Ahmed, T., Chowdhury, T. M., & Shatabda, S. (2023). FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation of Musculoskeletal Radiographs. Figshare. https://figshare.com/articles/dataset/The_dataset/22363012?file=43283628
  ```
- Research Paper:
  ```
  Ilyas, M. (2025). ResNet50-Driven Bone Fracture Detection with Attention-Augmented U-Net Segmentation.
  ```
