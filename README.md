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

### Validation (Freeze Backbone)

| Few-Shot | Epochs | LR   | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|----------|--------|------|-----------|--------|---------|--------------|
| 0        | –      | –    | 0.292     | 0.412  | 0.263   | 0.156        |
| 5        | 5      | 1e-5 | 0.317     | 0.471  | 0.317   | 0.188        |
| 10       | 8      | 1e-5 | 0.469     | 0.575  | 0.530   | 0.323        |
| 25       | 12     | 3e-5 | 0.570     | 0.713  | 0.605   | 0.367        |
| 50       | 15     | 5e-5 | 0.667     | 0.830  | 0.667   | 0.398        |

### Validation (No Backbone Freezing)

| Few-Shot | Epochs | LR   | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|----------|--------|------|-----------|--------|---------|--------------|
| 0        | –      | –    | 0.292     | 0.412  | 0.263   | 0.156        |
| 5        | 5      | 1e-5 | 0.430     | 0.579  | 0.430   | 0.258        |
| 10       | 8      | 1e-5 | 0.508     | 0.705  | 0.508   | 0.303        |
| 25       | 12     | 3e-5 | 0.669     | 0.817  | 0.669   | 0.379        |
| 50       | 15     | 5e-5 | 0.672     | 0.805  | 0.672   | 0.416        |


### Aquarium (Freeze backbone vs No-Freeze backbone)

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

### Floor Plans

### Validation (Freeze Backbone)

| Few-Shot | Epochs | LR   | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|----------|--------|------|-----------|--------|---------|--------------|
| 0        | –      | –    | 0.018     | 0.034  | 0.006   | 0.002        |
| 5        | 8      | 1e-5 | 0.092     | 0.115  | 0.092   | 0.058        |
| 10       | 8      | 1e-5 | 0.620     | 0.708  | 0.620   | 0.363        |
| 25       | 12     | 3e-5 | 0.793     | 0.861  | 0.793   | 0.485        |
| 50       | 15     | 5e-5 | 0.798     | 0.862  | 0.798   | 0.504        |

### Validation (No Backbone Freezing)

| Few-Shot | Epochs | LR   | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|----------|--------|------|-----------|--------|---------|--------------|
| 0        | –      | –    | 0.018     | 0.034  | 0.006   | 0.002        |
| 5        | 8      | 1e-5 | 0.162     | 0.178  | 0.162   | 0.101        |
| 10       | 8      | 1e-5 | 0.652     | 0.756  | 0.652   | 0.375        |
| 25       | 12     | 3e-5 | 0.804     | 0.869  | 0.804   | 0.494        |
| 50       | 15     | 5e-5 | 0.755     | 0.859  | 0.808   | 0.507        |

### Entrance 

### Validation (Freeze Backbone)

| Few-Shot | Epochs | LR   | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|----------|--------|------|-----------|--------|---------|--------------|
| 0        | –      | –    | 0.346     | 0.621  | 0.346   | 0.147        |
| 5        | 5      | 1e-5 | 0.490     | 0.517  | 0.490   | 0.234        |
| 10       | 8      | 1e-5 | 0.573     | 0.621  | 0.573   | 0.286        |
| 25       | 12     | 3e-5 | 0.724     | 0.793  | 0.724   | 0.387        |
| 50       | 15     | 5e-5 | 0.755     | 0.828  | 0.755   | 0.459        |

### Validation (No Backbone Freezing)

| Few-Shot | Epochs | LR   | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|----------|--------|------|-----------|--------|---------|--------------|
| 0        | –      | –    | 0.346     | 0.621  | 0.346   | 0.147        |
| 5        | 5      | 1e-5 | 0.469     | 0.483  | 0.469   | 0.223        |
| 10       | 8      | 1e-5 | 0.542     | 0.586  | 0.542   | 0.282        |
| 25       | 12     | 3e-5 | 0.616     | 0.724  | 0.616   | 0.366        |
| 50       | 15     | 5e-5 | 0.733     | 0.828  | 0.733   | 0.452        |


### Vertebra 

### Validation (Freeze Backbone)

| Few-Shot | Epochs | LR   | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|----------|--------|------|-----------|--------|---------|--------------|
| 0        | –      | –    | 0.000     | 0.000  | 0.000   | 0.000        |
| 5        | 5      | 1e-5 | 0.001     | 0.004  | 0.001   | 0.001        |
| 10       | 8      | 1e-5 | 0.635     | 0.731  | 0.635   | 0.291        |
| 25       | 12     | 3e-5 | 0.876     | 0.915  | 0.876   | 0.439        |
| 50       | 15     | 5e-5 | 0.906     | 0.934  | 0.906   | 0.461        |

### Validation (No Backbone Freezing)

| Few-Shot | Epochs | LR   | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|----------|--------|------|-----------|--------|---------|--------------|
| 0        | –      | –    | 0.000     | 0.000  | 0.000   | 0.000        |
| 5        | 8      | 1e-5 | 0.013     | 0.022  | 0.013   | 0.004        |
| 10       | 8      | 1e-5 | 0.684     | 0.790  | 0.684   | 0.298        |
| 25       | 12     | 3e-5 | 0.862     | 0.904  | 0.862   | 0.427        |
| 50       | 15     | 5e-5 | 0.912     | 0.946  | 0.912   | 0.472        |
