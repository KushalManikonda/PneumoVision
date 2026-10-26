# PneumoVision

PneumoVision is a deep learning project that uses a Convolutional Neural Network (CNN) to classify chest X-ray images into **NORMAL** and **PNEUMONIA** categories. The project covers dataset exploration, image preprocessing, CNN construction, model training, and evaluation using standard classification metrics.

**Notebook:** [View PneumoVision on Kaggle](https://www.kaggle.com/code/kushalman/pneumovision)

> **Important:** This project is an educational machine-learning prototype, not a clinically validated diagnostic system. Its predictions must not be used as a substitute for assessment by a qualified healthcare professional.

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Technologies and Libraries](#technologies-and-libraries)
- [Workflow](#workflow)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Evaluation Details](#evaluation-details)
- [How to Run](#how-to-run)
- [Limitations and Future Work](#limitations-and-future-work)
- [Repository Structure](#repository-structure)
- [Acknowledgements](#acknowledgements)

## Overview

Pneumonia is a respiratory condition that can cause inflammation in the lungs. Chest X-rays are commonly used by healthcare professionals as part of the assessment process. PneumoVision explores how a CNN can learn visual patterns in chest X-ray images and classify them into two categories:

- **NORMAL** — the image is labelled as normal in the dataset.
- **PNEUMONIA** — the image is labelled as pneumonia in the dataset.

The notebook implements the image-classification pipeline with TensorFlow and Keras and evaluates predictions using accuracy, precision, recall, F1-score, and a confusion matrix.

## Objectives

- Prepare and explore a chest X-ray image dataset.
- Resize and batch images for CNN input.
- Build and train a CNN for binary image classification.
- Track training and validation loss and accuracy.
- Evaluate predictions using a confusion matrix and classification report.
- Document the model's performance and limitations.

## Dataset

The notebook uses the **Chest X-Ray Images (Pneumonia)** dataset associated with Paul Mooney.

- **Dataset source:** [Chest X-Ray Images (Pneumonia) on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes:** `NORMAL` and `PNEUMONIA`
- **Model input size:** `224 × 224` pixels
- **Color mode:** RGB (3 channels)

The notebook reads the dataset from Kaggle input directories. The final data generators contain:

| Split used by the model | Images |
|---|---:|
| Training | 4,172 |
| Validation | 522 |
| Test | 522 |
| **Total used in these generators** | **5,216** |

**Data-split note:** In the current notebook, `train_df`, `valid_df`, and `test_df` are created by splitting the image paths collected from the dataset's `train` directory using an 80/20 split followed by an even split of the remaining 20% (`random_state=42`). The notebook also builds dataframes from the dataset's original `val` and `test` directories, but those dataframes are not used to create the final model generators. Therefore, the metrics below are from the notebook's 522-image test split, not from the dataset's original official test directory. This should be corrected if a strictly independent evaluation on the official test set is required.

## Technologies and Libraries

The notebook uses:

- **Python**
- **TensorFlow / Keras** — neural-network construction and training
- **NumPy** — numerical operations
- **Pandas** — dataset metadata and image-path tables
- **Matplotlib** — plots and visualizations
- **Seaborn** — visualization styling
- **Plotly Express** — visualization library imported in the notebook
- **OpenCV (`cv2`) and Pillow (`PIL`)** — image-related utilities imported in the notebook
- **scikit-learn** — data splitting, confusion matrix, and classification report

## Workflow

1. **Load the data:** Read image paths and labels from the dataset directories.
2. **Explore the data:** Inspect image samples and class distribution.
3. **Prepare the data:** Resize images to `224 × 224`, use RGB input, and create batches with Keras `ImageDataGenerator`.
4. **Split the data:** Split the collected image paths into training, validation, and test subsets using `train_test_split` and `random_state=42`.
5. **Build the CNN:** Stack convolutional and max-pooling layers, followed by dense classification layers.
6. **Train the model:** Train for six epochs and monitor validation performance.
7. **Evaluate:** Calculate loss and accuracy, plot training history, and generate a confusion matrix and classification report.

## Model Architecture

The model is a sequential CNN built with Keras. It uses convolutional layers with ReLU activations to extract image features, max-pooling layers to reduce spatial dimensions, and dense layers to perform classification.

| Component | Configuration |
|---|---|
| Input | `224 × 224 × 3` |
| Convolutional blocks | 2 × 64 filters, 2 × 128 filters, 3 × 256 filters, 3 × 512 filters, and 3 × 512 filters |
| Convolution kernel | `3 × 3`, same padding, ReLU activation |
| Pooling | `2 × 2` max pooling after each convolutional block |
| Flatten | Converts feature maps into a vector |
| Dense layers | 256 units, then 64 units, ReLU activation |
| Output | 2 units with softmax activation |
| Total parameters | 21,154,050 |
| Optimizer | Adamax |
| Learning rate | 0.001 |
| Loss function | Categorical cross-entropy |
| Metric | Accuracy |

## Training Configuration

| Parameter | Value |
|---|---|
| Image dimensions | `224 × 224` |
| Batch size | 16 |
| Epochs | 6 |
| Training images | 4,172 |
| Validation images | 522 |
| Test images | 522 |
| Input channels | 3 (RGB) |
| Training framework | TensorFlow / Keras |

## Results

The following results are taken from the executed notebook.

| Metric | Loss | Accuracy |
|---|---:|---:|
| Training | 0.0329 | 98.59% |
| Validation | 0.0780 | 97.70% |
| Test | 0.1131 | 96.55% |

### Test classification report

The classification report was generated on 522 test-split images.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| NORMAL | 0.94 | 0.92 | 0.93 | 133 |
| PNEUMONIA | 0.97 | 0.98 | 0.98 | 389 |
| **Macro average** | **0.96** | **0.95** | **0.95** | **522** |
| **Weighted average** | **0.97** | **0.97** | **0.97** | **522** |
| **Accuracy** |  |  | **0.97** | **522** |

The test confusion matrix in the notebook is:

| Actual \ Predicted | NORMAL | PNEUMONIA |
|---|---:|---:|
| NORMAL | 123 | 10 |
| PNEUMONIA | 8 | 381 |

These metrics describe performance on the split used by the current notebook. They should not be interpreted as evidence of clinical readiness or generalization to images from other hospitals or patient populations.

## How to Run

### Option 1: Run on Kaggle

1. Open the [PneumoVision Kaggle notebook](https://www.kaggle.com/code/kushalman/pneumovision).
2. Ensure the Chest X-Ray Images (Pneumonia) dataset is attached as notebook input.
3. Select an available accelerator if needed.
4. Run the notebook cells in order.

### Option 2: Run locally

1. Clone this repository.
2. Install the dependencies listed in `requirements.txt` (if provided).
3. Download the dataset from the source linked above and update the dataset paths in the notebook for your local environment.
4. Open `PneumoVision.ipynb` in Jupyter Notebook or JupyterLab and run the cells in order.

The notebook currently uses Kaggle-specific filesystem paths, so those paths must be changed before running locally. Exact dependency versions have not been specified in the notebook; record tested versions in `requirements.txt` for reproducibility.

## Limitations and Future Work

- **Independent evaluation:** Update the data pipeline to use the dataset's original train, validation, and test directories consistently, and evaluate once on the untouched official test set.
- **Class imbalance:** The classes are not equally represented in the evaluated split. Consider class weighting or other appropriate techniques and report per-class metrics.
- **Generalization:** Evaluate on an external dataset from a different source before making claims about performance outside this dataset.
- **Explainability:** Add methods such as Grad-CAM to investigate which image regions influence predictions.
- **Model reliability:** Assess calibration, subgroup performance, image quality issues, and false-negative cases.
- **Clinical validation:** Any clinical use would require appropriate expert review, rigorous validation, and applicable regulatory and ethical processes.

## Repository Structure

A suggested repository layout is:

```text
PneumoVision/
├── README.md
├── PneumoVision.ipynb
├── requirements.txt
├── assets/
│   └── training-results.png
├── results/
│   ├── confusion-matrix.png
│   └── training-curves.png
└── models/
    └── README.md
```

Only add files and folders that are actually present. The dataset itself is not included in this repository; follow its source's terms and instructions to obtain it.

## Acknowledgements

- **Dataset:** [Chest X-Ray Images (Pneumonia) — Paul Mooney on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Notebook:** [PneumoVision on Kaggle](https://www.kaggle.com/code/kushalman/pneumovision)
- **Frameworks:** TensorFlow, Keras, and scikit-learn.

---

**Disclaimer:** PneumoVision is an educational research project. It is not intended for diagnosis, treatment decisions, or other clinical use.
