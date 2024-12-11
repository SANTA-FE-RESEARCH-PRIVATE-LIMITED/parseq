import glob
import os
import random

import cv2
import tqdm

ROOT_DIR = "data/Hindi-HW"
CACHE_DIR = "data/Hindi-HW/cache"
NUM_SAMPLES = 1000


def crop_image_from_contour(image_path: str, padding: int = 10):
    """
    Crops an image based on the largest contour detected using binarization.

    Args:
        - image_path (str): Path to the input image.
        - padding (int): Padding around the bounding box.
    Returns:
        - numpy.ndarray: The cropped image.
    """
    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 100  # Ignore small contours
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + 2 * padding, original_image.shape[1] - x)
    h = min(h + 2 * padding, original_image.shape[0] - y)
    cropped_image = original_image[y : y + h, x : x + w]
    return cropped_image


if __name__ == "__main__":
    files = [file for file in glob.iglob(os.path.join(ROOT_DIR, "**/*.jpg"), recursive=True)]
    random.shuffle(files)
    files = files[:NUM_SAMPLES]
    for file in tqdm.tqdm(files, desc="Cropping images"):
        cache_file = os.path.join(CACHE_DIR, os.path.relpath(file, ROOT_DIR))
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        cropped_image = crop_image_from_contour(file, 40)
        cv2.imwrite(cache_file, cropped_image)
