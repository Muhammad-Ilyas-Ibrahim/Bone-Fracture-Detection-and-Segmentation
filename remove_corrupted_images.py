import os
import pandas as pd

# Path to the corrupted images list
corrupted_images_file = 'corrupted_images.txt'
# Path to the directory containing the images
images_dir = os.path.join('images', 'Non_fractured')
# Path to the dataset CSV file
dataset_csv_file = 'dataset.csv'

def main():
    # Read the list of corrupted image filenames
    with open(corrupted_images_file, 'r') as f:
        corrupted_images = [line.strip() for line in f if line.strip()]

    deleted = []
    not_found = []

    for img_name in corrupted_images:
        img_path = os.path.join(images_dir, img_name)
        if os.path.exists(img_path):
            try:
                os.remove(img_path)
                print(img_path, 'deleted')
                deleted.append(img_name)
            except Exception as e:
                print(f"Error deleting {img_name}: {e}")
        else:
            not_found.append(img_name)

    print(f"Deleted {len(deleted)} images.")
    if not_found:
        print(f"{len(not_found)} images not found in {images_dir}:")
        for img in not_found:
            print(f"  {img}")

    # Remove records from dataset.csv
    if os.path.exists(dataset_csv_file):
        df = pd.read_csv(dataset_csv_file)
        initial_count = len(df)
        df = df[~df['image_id'].isin(corrupted_images)]
        removed_count = initial_count - len(df)
        df.to_csv(dataset_csv_file, index=False)
        print(f"Removed {removed_count} records from {dataset_csv_file}.")
    else:
        print(f"{dataset_csv_file} not found. Skipped CSV update.")

if __name__ == "__main__":
    main()