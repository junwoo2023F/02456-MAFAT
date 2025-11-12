import pandas as pd
import os
import cv2
import numpy as np
from tqdm import tqdm


# ===== CONFIG =====
test_csv = './dataset_v2/test.csv'
image_root = './dataset_v2/root/test'
output_root = './blurred_test_outputs'


kernel_sizes = [0, 11, 21, 31, 41]   # 0 = original
level_names = ['Original', 'Blur1', 'Blur2', 'Blur3', 'Blur4']


# ===== FUNCTIONS =====
def crop_image(image, coords):
    """Crop image based on 8 coordinates, clipping to image bounds."""
    h, w = image.shape[:2]
    x_coords = coords[::2]
    y_coords = coords[1::2]
    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))


    # Clip to image dimensions
    x_min, x_max = max(0, x_min), min(w, x_max)
    y_min, y_max = max(0, y_min), min(h, y_max)


    # Ensure non-empty crop
    if x_max <= x_min or y_max <= y_min:
        return None


    return image[y_min:y_max, x_min:x_max]


# ===== MAIN FUNCTION =====
def main():
    # Load CSV
    df = pd.read_csv(test_csv)
    df.columns = df.columns.str.strip()  # remove any whitespace


    # Coordinate columns
    coord_cols = ['p1_x', 'p_1y', 'p2_x', 'p2_y', 'p3_x', 'p3_y', 'p4_x', 'p4_y']


    # Check columns exist
    for col in coord_cols:
        if col not in df.columns:
            raise ValueError(f"git aCSV missing column: {col}")


    # Create output folders
