## 1. Introduction

In traditional computer vision workflows, dataset annotation is often the most time-consuming step. Manually drawing bounding boxes for thousands of images quickly becomes a bottleneck, especially for domain-specific objects where zero shot may not generalize well.

Although modern vision models are trained on large-scale data and many works have explored zero-shot detection to reduce labeling effort, in practice the recall is often too low for reliable self-annotation. Missed objects and excessive false positives still require substantial human correction, limiting the usefulness of purely zero-shot approaches.

By introducing a small amount of labeled data, few-shot learning can improve detection recall and provide better pre-annotations. This project studies a recall-oriented few-shot approach with human-in-the-loop as Stage 1 of an annotation pipeline. The refined annotations can then be used to train a fully supervised detector (e.g., YOLO or RF-DETR) as Stage 2 for improved final performance.

## 2. Dataset

### 1) [Indoor Fire](https://www.kaggle.com/datasets/pengbo00/home-fire-dataset)

Classes: Fire, Smoke 

Goal: Detect small fire regions in indoor environments for early fire detection.

### 2) [Aquarium](https://public.roboflow.com/object-detection/aquarium/2)

Classes : Jellyfish, Pengiun, Shark, Starfish, Stingary  

Goal: Detect and distinguish aquatic species for educational and automatic counting applications.

### 3) [Floor Plans](https://www.kaggle.com/datasets/umairinayat/floor-plans-500-annotated-object-detection?select=data.yaml)

Classes : Door symbol, Window symbol, Zone  

Goal: Detect architectural symbols for automatic CAD conversion (AutoCAD) and object counting.

### 4) [Entrance](https://www.kaggle.com/datasets/evanrantala/entrances-dataset-street-level-object-detection)

Classes : Entrance 

Goal: Detect building entrances in street scenes for navigation and delivery applications.

### 5) [Vertebra](https://www.kaggle.com/datasets/salmankey/scoliosis-yolov5-annotated-spine-x-ray-dataset)

Classes : Vertebra 

Goal: Detect individual vertebrae to support Cobb angle estimation and scoliosis assessment.

## 3. Method

We use Grounding DINO as the base detector for few-shot self-annotation due to its open-vocabulary detection capability. The goal is to improve detection recall with a small amount of labeled data while keeping human correction effort low.

### Few-Shot Setup
For each dataset, we sample 5, 10, 25, and 50 labeled images per class as few-shot training data. These samples are used to fine-tune the model and adapt it to domain-specific objects.

### Backbone Adaptation
We study two training strategies: freezing the backbone and fine-tuning the backbone. This comparison allows us to analyze whether adapting feature representations improves recall in low-data settings.

### Human-in-the-Loop
The trained model is applied to unlabeled images to generate pre-annotations. A human annotator then reviews the predictions to remove false positives and correct missed detections. Although few-shot learning improves recall, human verification is still required to ensure annotation quality.

### Evaluation
Recall is used as the primary evaluation metric, as missing objects directly increase manual correction effort in self-annotation scenarios. We report recall under different shot settings to analyze reliability across datasets.

## 4. Result

### Indoor Fire (Freeze backbone vs No-Freeze backbone)

| Few-Shot | Epochs | LR   | Recall (Freeze) | Recall (No-Freeze) |
|----------|--------|------|----------------|-------------------|
| 0        | –      | –    | 0.412           | 0.412             |
| 5        | 5      | 1e-5 | 0.471           | 0.579             |
| 10       | 8      | 1e-5 | 0.575           | 0.705             |
| 25       | 12     | 3e-5 | 0.713           | 0.817             |
| 50       | 15     | 5e-5 | **0.830**           | 0.805             |

