# Image Similarity Search using ResNet and LAB Color Features

This project compares images based on visual similarity using deep features extracted from pretrained ResNet models (ResNet50, ResNet101, ResNet152) and LAB color space statistics.

It is designed to identify similar images from a dataset given one or more input images. Feature comparison is done using cosine similarity and LAB color distance.

---

## 📌 Features

- Extracts feature vectors using pretrained ResNet models
- Computes average LAB color and includes it in similarity scoring
- Combines structural (deep) and color-based similarity
- Supports batch processing of large image datasets
- Results saved with similarity score and matched file information

---

## 🧱 Models Compared

| Model       | Layers | Params (M) | Accuracy (Top-1) | Speed  | Usage                              |
|-------------|--------|------------|------------------|--------|------------------------------------|
| ResNet50    | 50     | ~25M       | ~76.0%           | Fast   | Ideal for lightweight applications |
| ResNet101   | 101    | ~44M       | ~77.4%           | Medium | Balanced depth and performance     |
| ResNet152   | 152    | ~60M       | ~78.3%           | Slower | Best feature quality, higher cost  |

---

**Summary:**  
- ResNet50 offers a good speed-accuracy trade-off for most applications.  
- ResNet101 balances deeper architecture with reasonable computational cost.  
- ResNet152 achieves highest accuracy but requires significant computation and memory.
