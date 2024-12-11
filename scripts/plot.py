import os

import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from matplotlib import rcParams
from PIL import Image

rcParams["font.family"] = ["Noto Sans", "Noto Sans Devanagari"]
rcParams["font.size"] = 20


def load_data(result_root, image_file, prediction_file):
    """
    Load image and prediction data from CSV files.

    Args:
        - result_root: Root directory containing the prediction CSV files.
        - image_file: Name of the CSV file containing image data.
        - prediction_file: Name of the CSV file containing prediction data.

    Returns:
        - Merged DataFrame containing both image and prediction data.
    """
    images_csv = os.path.join(result_root, image_file)
    predictions_csv = os.path.join(result_root, prediction_file)

    images_df = pd.read_csv(images_csv)
    predictions_df = pd.read_csv(predictions_csv)
    merged_df = pd.concat([images_df, predictions_df], axis=1)
    return merged_df


def create_directories(result_root):
    """
    Create directories for saving correct and incorrect prediction images.

    Args:
        - result_root: Root directory where the result directories will be created.
    """
    os.makedirs(os.path.join(result_root, "correct"), exist_ok=True)
    os.makedirs(os.path.join(result_root, "incorrect"), exist_ok=True)


def plot_image_prediction(row, image_root, result_root, index):
    """
    Plot the original image and prediction details, and save the plot.

    Args:
        - row: DataFrame row containing image.
        - image_root: Root directory containing the images.
        - result_root: Root directory where the result images will be saved.
        - index: Index of the current row.
    """
    try:
        img = Image.open(os.path.join(image_root, row["image"]))
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title("Original Image")
        plt.subplot(1, 2, 2)
        plt.text(
            0.5,
            0.7,
            f"True Label: {row['label']}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.text(
            0.5,
            0.5,
            f"Predicted: {row['prediction']}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.text(
            0.5,
            0.3,
            f"Correct: {row['correct']}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.axis("off")
        plt.title("Prediction Details")
        plt.tight_layout()
        if row["correct"]:
            save_path = os.path.join(result_root, f"correct/{index}.png")
        else:
            save_path = os.path.join(result_root, f"incorrect/{index}.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error processing image at index {index}: {e}")


def create_image_prediction_plots(image_root, result_root, image_file, prediction_file):
    """
    Create plots for image predictions and save them in the specified directories.

    Args:
        - image_root: Root directory containing the image.
        - result_root: Root directory where the result images will be saved.
        - image_file: Name of the CSV file containing image data.
        - prediction_file: Name of the CSV file containing prediction data.
    """
    merged_df = load_data(result_root, image_file, prediction_file)
    create_directories(result_root)

    for index, row in tqdm.tqdm(merged_df.iterrows(), desc="Plotting", total=len(merged_df)):
        plot_image_prediction(row, image_root, result_root, index)


if __name__ == "__main__":
    create_image_prediction_plots(
        image_root="data/Hindi",
        result_root="indic-models/parseq/hindi/run_01/pretrained/results",
        image_file="Hindi_images.csv",
        prediction_file="Hindi_predictions.csv",
    )
