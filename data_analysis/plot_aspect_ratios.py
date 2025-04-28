import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def calculate_aspect_ratios(image_dir_train, image_dir_valid):
    def plot_aspect_ratios(aspect_ratios, title):
        counts, bins, patches = plt.hist(aspect_ratios, bins=20, edgecolor='black')
        plt.title(title)
        plt.xlabel('Aspect Ratio')
        plt.ylabel('Frequency')

        # Add text labels next to each bar (horizontally)
        for count, bin_start, bin_end in zip(counts, bins[:-1], bins[1:]):
            if count > 0:
                bin_center = (bin_start + bin_end) / 2
                plt.text(bin_center, count/2, f'{bin_center:.2f}', ha='left', va='center', fontsize=8, rotation=0)

        plt.show()

    # --- Training set ---
    aspect_ratios_train = []
    for image in os.listdir(image_dir_train):
        img_path = os.path.join(image_dir_train, image)
        try:
            img = Image.open(img_path)
            width, height = img.size
            if height != 0:
                aspect_ratios_train.append(width / height)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    plot_aspect_ratios(aspect_ratios_train, 'Aspect Ratios of Training Images Histogram')

    # --- Validation set ---
    aspect_ratios_valid = []
    for image in os.listdir(image_dir_valid):
        img_path = os.path.join(image_dir_valid, image)
        try:
            img = Image.open(img_path)
            width, height = img.size
            if height != 0:
                aspect_ratios_valid.append(width / height)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    plot_aspect_ratios(aspect_ratios_valid, 'Aspect Ratios of Validation Images Histogram')

# Run
calculate_aspect_ratios(
    "/Users/andre/projects/MiniProjectTDT4265/datasets/rgb/images/train",
    "/Users/andre/projects/MiniProjectTDT4265/datasets/rgb/images/valid"
)
