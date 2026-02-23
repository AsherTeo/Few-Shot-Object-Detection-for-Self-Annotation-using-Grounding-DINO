## 1. Introduction

In traditional computer vision workflows, dataset annotation is often the most time-consuming and labor-intensive step. Anyone who has manually drawn bounding boxes knows how tedious it becomes when working with thousands or even millions of images. This challenge is even greater for domain-specific objects, where zero-shot detection models may not generalize well.

Even though modern vision models are trained on millions of images and have seen a wide variety of objects, and many works have demonstrated zero-shot detection as a way to ease daily annotation work, in practice the recall is often too low for reliable self-annotation. Missed objects and excessive false positives still require substantial human effort to correct, which limits the usefulness of purely zero-shot approaches.

By introducing a small amount of labeled data, few-shot learning can significantly improve detection recall, making model predictions more suitable as pre-annotations. However, few-shot detection does not remove the need for human involvement. Instead, it reduces the amount of manual correction required by improving the quality of automatic predictions.

In this project, we explore a recall-oriented few-shot approach for self-annotation with human-in-the-loop. We evaluate few-shot object detection using 5, 10, 25, and 50 samples per class across six datasets, and analyze how detection recall changes under different shot settings. We also study the effect of freezing versus fine-tuning the backbone to identify configurations that provide more reliable pre-annotations for practical annotation pipelines.
