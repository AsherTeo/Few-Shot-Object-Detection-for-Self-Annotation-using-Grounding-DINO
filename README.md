**1) Introduction**

Dataset annotation is a costly and time-consuming process, particularly when labeling thousands or millions of images.
This challenge becomes more severe for domain-specific objects, where zero-shot object detection models often struggle to generalize reliably.

Even when zero-shot detection achieves reasonable recall, further improving recall is crucial for self-annotation, as missed objects directly increase human correction effort. A recall-oriented few-shot approach can therefore reduce annotation time and allow practitioners to focus more on optimizing the final detection model.

In this project, we investigate few-shot object detection for self-annotation using 5, 10, 25, and 50 samples per class across six datasets. We analyze how detection recall changes under different shot settings and examine the impact of freezing versus fine-tuning the backbone to identify reliable configurations for practical self-annotation.
