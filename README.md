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

### Aquarium

| Few-Shot | Epochs | LR   | Recall (Freeze) | Recall (No-Freeze) |
|----------|--------|------|----------------|-------------------|
| 0        | –      | –    | 0.346           | 0.346             |
| 5        | 5      | 1e-5 | 0.604           | 0.629             |
| 10       | 8      | 1e-5 | 0.643           | 0.636             |
| 25       | 12     | 3e-5 | 0.809           | 0.827             |
| 50       | 15     | 5e-5 | **0.879**      | 0.818             |

### Floor Plans

| Zero-shot | 5-shot (No-Freeze) | 25-shot (No-Freeze, best) |
|-----------|--------------------|---------------------------|
| ![](https://github.com/user-attachments/assets/df079dad-0534-498e-980f-e977c927e03f) | ![](https://github.com/user-attachments/assets/bf6936c4-c88b-4eab-b6c0-2c22f73df6b8) | ![](https://github.com/user-attachments/assets/216f2924-2a2b-41f6-95c2-c2fcfa94d42d) |

| Few-Shot | Epochs | LR   | Recall (Freeze) | Recall (No-Freeze) |
|----------|--------|------|----------------|-------------------|
| 0        | –      | –    | 0.034           | 0.034             |
| 5        | 8      | 1e-5 | 0.115           | 0.178             |
| 10       | 8      | 1e-5 | 0.708           | 0.756             |
| 25       | 12     | 3e-5 | 0.861           | **0.869**             |
| 50       | 15     | 5e-5 | 0.862           | 0.859        |



### Entrance

| Few-Shot | Epochs | LR   | Recall (Freeze) | Recall (No-Freeze) |
|----------|--------|------|----------------|-------------------|
| 0        | –      | –    | 0.621           | 0.621             |
| 5        | 5      | 1e-5 | 0.517           | 0.483             |
| 10       | 8      | 1e-5 | 0.621           | 0.586             |
| 25       | 12     | 3e-5 | 0.793           | 0.724             |
| 50       | 15     | 5e-5 | **0.828**      | 0.828             |


### Vertebra

| Few-Shot | Epochs | LR   | Recall (Freeze) | Recall (No-Freeze) |
|----------|--------|------|----------------|-------------------|
| 0        | –      | –    | 0.000           | 0.000             |
| 5        | 5      | 1e-5 | 0.004           | 0.022             |
| 10       | 8      | 1e-5 | 0.731           | 0.790             |
| 25       | 12     | 3e-5 | 0.915           | 0.904             |
| 50       | 15     | 5e-5 | 0.934           | **0.946**         |

## 5. Insight

From experiments on five datasets, we observe clear trends between shot size and backbone strategy.

For 5-shot and 10-shot, unfreezing the backbone usually gives better recall, since the model needs to adapt its features to new domains with very limited data.

For 25-shot, unfreezing the backbone is still often better, as moderate data allows useful domain adaptation.

For 50-shot, the best strategy depends on zero-shot performance:
- If zero-shot recall ≥ 0.3, freezing the backbone works better because the pretrained features are already useful and freezing improves stability.
- If zero-shot recall is very low, unfreezing the backbone is needed to learn domain-specific features.
  
## 6. Conclusion

- Datasets with some zero-shot knowledge benefit most from 50-shot training with a frozen backbone.

- Datasets with little or no zero-shot knowledge benefit most from 50-shot training with an unfrozen backbone.

- Very small few-shot sizes (5 shots) give limited improvement and are not sufficient for reliable self-annotation.

- Although 25 shots per class is often a good balance between effort and performance, the best few-shot size still depends on the dataset and task.

**Overall, 25-shot training is often the best practical choice, as it provides a good balance between annotation effort, dataset size, and recall improvement.**
