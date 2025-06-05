import os
import shutil
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet152_Weights
from PIL import Image
from skimage import color
import numpy as np

# --- Device configuration (use GPU if available) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Model and transform definition ---
weights = ResNet152_Weights.DEFAULT
model = models.resnet152(weights=weights)
model = nn.Sequential(*list(model.children())[:-1])  # remove classification layer
model.eval()

preprocess = weights.transforms()

# --- Feature extraction ---
def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image).squeeze().numpy()
    return features

# --- LAB average color extraction ---
def get_average_lab_color(image_path):
    image = Image.open(image_path).convert('RGB').resize((224, 224))
    lab_image = color.rgb2lab(np.array(image))
    avg_lab = lab_image.reshape(-1, 3).mean(axis=0)
    return avg_lab

def lab_distance(lab1, lab2):
    return np.linalg.norm(lab1 - lab2)

# --- Cosine similarity ---
def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot / norm if norm > 0 else 0

# --- Combined score calculation ---
def combined_similarity_score(resnet_score, lab_dist, lab_max=100):
    lab_score = 1 - (lab_dist / lab_max)
    lab_score = max(min(lab_score, 1), 0)
    return 0.7 * resnet_score + 0.3 * lab_score

# --- Get all images from dataset ---
def get_all_images(root_dir):
    return [os.path.join(dp, f)
            for dp, dn, filenames in os.walk(root_dir)
            for f in filenames if f.lower().endswith('.jpg')]

# --- Generate a unique file name ---
def get_unique_filename(directory, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return new_filename

# --- Main function ---
def find_similar_images(input_image_paths, dataset_root, output_dir, threshold=0.80):
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "results.txt")

    input_features = []
    for path in input_image_paths:
        try:
            feat = extract_features(path)
            avg_lab = get_average_lab_color(path)
            input_features.append((path, feat, avg_lab))
        except Exception as e:
            print(f"Failed to read input image: {path} - {e}")

    all_images = get_all_images(dataset_root)

    with open(results_file, 'w', encoding='utf-8') as f:
        for img_path in tqdm(all_images, desc="Comparing"):
            try:
                target_feat = extract_features(img_path)
                target_lab = get_average_lab_color(img_path)

                for input_path, input_feat, input_lab in input_features:
                    resnet_score = cosine_similarity(input_feat, target_feat)
                    lab_dist = lab_distance(input_lab, target_lab)
                    final_score = combined_similarity_score(resnet_score, lab_dist)

                    if final_score >= threshold:
                        input_name = Path(input_path).stem
                        target_name = Path(img_path).stem
                        target_ext = Path(img_path).suffix
                        new_filename = f"match_{input_name}_{target_name}{target_ext}"

                        # Ensure unique file name
                        new_filename = get_unique_filename(output_dir, new_filename)

                        dst_path = os.path.join(output_dir, new_filename)
                        shutil.copy(img_path, dst_path)

                        f.write(f"{new_filename} - %{final_score * 100:.2f} (match: {os.path.basename(input_path)})\n")
                        break
            except Exception as e:
                print(f"Error: {img_path} - {e}")

# --- Usage ---
if __name__ == "__main__":
    input_images = [
        r"D:\",
        r"D:\"
    ]
    dataset_path = r"D:\"
    output_path = r"D:\"

    find_similar_images(input_images, dataset_path, output_path, threshold=0.80)
