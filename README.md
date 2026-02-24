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

### Fire & Smoke (Freeze backbone vs No-Freeze backbone)

### Validation (Freeze Backbone)

| Few-Shot | Epochs | LR   | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|----------|--------|------|-----------|--------|---------|--------------|
| 0        | –      | –    | 0.217     | 0.346  | 0.217   | 0.123        |
| 5        | 5      | 1e-5 | 0.489     | 0.604  | 0.489   | 0.322        |
| 10       | 8      | 1e-5 | 0.557     | 0.643  | 0.557   | 0.350        |
| 25       | 12     | 3e-5 | 0.744     | 0.809  | 0.744   | 0.527        |
| 50       | 15     | 5e-5 | 0.856     | 0.879  | 0.770   | 0.536        |

### Validation (No Backbone Freezing)

| Few-Shot | Epochs | LR   | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|----------|--------|------|-----------|--------|---------|--------------|
| 0        | –      | –    | 0.217     | 0.346  | 0.217   | 0.123        |
| 5        | 5      | 1e-5 | 0.495     | 0.629  | 0.495   | 0.309        |
| 10       | 8      | 1e-5 | 0.622     | 0.636  | 0.615   | 0.409        |
| 25       | 12     | 3e-5 | 0.753     | 0.827  | 0.753   | 0.513        |
| 50       | 15     | 5e-5 | 0.796     | 0.818  | 0.734   | 0.517        |


### Aquarium (Freeze backbone vs No-Freeze backbone)








