## 1. Introduction

In traditional computer vision workflows, dataset annotation is often the most time-consuming step. Manually drawing bounding boxes for thousands of images quickly becomes a bottleneck, especially for domain-specific objects where zero shot may not generalize well.

Although modern vision models are trained on large-scale data and many works have explored zero-shot detection to reduce labeling effort, in practice the recall is often too low for reliable self-annotation. Missed objects and excessive false positives still require substantial human correction, limiting the usefulness of purely zero-shot approaches.

By introducing a small amount of labeled data, few-shot learning can improve detection recall and provide better pre-annotations. This project studies a recall-oriented few-shot approach with human-in-the-loop as Stage 1 of an annotation pipeline. The refined annotations can then be used to train a fully supervised detector (e.g., YOLO or RF-DETR) as Stage 2 for improved final performance.

## 2. Dataset

### Indoor Fire

Classes: Fire, Smoke

Source: https://www.kaggle.com/datasets/pengbo00/home-fire-dataset

This dataset contains approximately 3.9k training images of indoor fire and smoke scenes. The goal is to detect small fire regions in indoor environments, where early detection is critical. 

### Aquarium


### Floor Plans



### Entrance



### Vertebra
