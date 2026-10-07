# Weather Condition Classification

Multi-class weather image classification using fine-tuned VGG-19, ResNet50, and EfficientNetB0 with Grad-CAM interpretability.

## Overview

This project investigates deep learning approaches for classifying weather images into eleven different weather conditions. Three ImageNet-pretrained convolutional neural network architectures — VGG-19, ResNet50, and EfficientNetB0 — were trained using a common transfer-learning and fine-tuning pipeline.

The models were evaluated on an independent test set, and Grad-CAM was used to provide visual explanations of model predictions.

## Weather Classes

The dataset contains 6,862 images belonging to 11 weather categories:

- Dew
- Fogsmog
- Frost
- Glaze
- Hail
- Lightning
- Rain
- Rainbow
- Rime
- Sandstorm
- Snow

## Dataset Split

The dataset was divided into three subsets:

| Split | Images | Percentage |
|---|---:|---:|
| Training | 4,798 | 70% |
| Validation | 1,031 | 15% |
| Independent Test | 1,033 | 15% |
| **Total** | **6,862** | **100%** |

The images were resized to 224 × 224 pixels. Data augmentation was applied to the training set.

## Models

The following ImageNet-pretrained architectures were evaluated:

- **VGG-19**
- **ResNet50**
- **EfficientNetB0**

A common classification head consisting of Global Average Pooling, a 128-neuron ReLU layer, dropout, and an 11-class softmax output was used for the models.

Training consisted of frozen-backbone training followed by selective fine-tuning.

## Results

Performance was measured on the independent test set.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| VGG-19 | 73.09% | 78.12% | 70.38% | 70.88% |
| ResNet50 | 89.55% | 90.53% | 90.04% | 90.15% |
| **EfficientNetB0** | **90.61%** | **91.44%** | **91.21%** | **91.29%** |

EfficientNetB0 achieved the best performance on the independent test set, with an accuracy of **90.61%** and a macro F1-score of **91.29%**.

## Error Analysis

The main classification errors occurred among visually similar weather categories.

In particular, confusion was observed between:

- Glaze and rime
- Frost and glaze
- Frost and rime
- Snow and glaze/rime
- Rain and snow

These errors indicate that visually similar weather conditions remain challenging for image-based classification.

## Grad-CAM Interpretability

Grad-CAM was used to visualise image regions contributing to model predictions.

This provides an interpretable view of the features used by the trained models when making weather classification decisions.

## Project Structure

```text
weather-condition-classification/
│
├── notebooks/
│   ├── README.md
│   ├── dl_weather_vgg.ipynb
│   ├── dl_weather_resnet.ipynb
│   └── dl_whether_efficientnet.ipynb
│
├── src/
│   └── README.md
│
├── results/
│   ├── figures/
│   └── metrics/
│
├── paper/
│   └── figures/
│
├── docs/
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
