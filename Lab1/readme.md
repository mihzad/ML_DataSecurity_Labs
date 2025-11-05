# Lab1: test Noise and FGSM- attacks on imagenet-pretrained architecture using ImageNet-10 dataset.
https://www.kaggle.com/datasets/liusha249/imagenet10

The architecture chosen was MobileNetV3-Large. the model outputs 1000 classes, so the mapping was used - new "others" class is added to the 10 available in dataset. if the class predicted is not one of the ten selected - its automatically passed as "others" into prediction data and confusion matrices.

## Base capabilities:
<img width="1201" height="920" alt="image" src="https://github.com/user-attachments/assets/dcd07d4b-e556-47cd-9f32-2245f2c02429" />
log:

performing testing...

| Class Name        | Precision | Recall | F1-Score | Support |
|--------------------|------------|---------|-----------|----------|
| king penguin       | 1.00 | 0.99 | 0.99 | 1300 |
| Maltese dog        | 1.00 | 0.91 | 0.95 | 1300 |
| snow leopard       | 1.00 | 0.97 | 0.99 | 1300 |
| airliner           | 0.99 | 0.93 | 0.96 | 1300 |
| airship            | 0.99 | 0.96 | 0.97 | 1300 |
| container ship     | 1.00 | 0.94 | 0.97 | 1300 |
| soccer ball        | 1.00 | 0.89 | 0.94 | 1300 |
| sports car         | 1.00 | 0.73 | 0.84 | 1300 |
| trailer truck      | 1.00 | 0.82 | 0.90 | 1300 |
| orange             | 1.00 | 0.86 | 0.93 | 1300 |
| others             | 0.00 | 0.00 | 0.00 | 0 |
| | | | |
| **Accuracy**       | — | — | **0.90** | 13000 |
| **Macro avg**      | 0.91 | 0.82 | 0.86 | 13000 |
| **Weighted avg**   | 1.00 | 0.90 | 0.94 | 13000 |


## Noise Attacks
### Random noise with epsilon = 5%
<img width="1139" height="886" alt="image" src="https://github.com/user-attachments/assets/f76eb018-f091-4d43-9c18-fe274f178c76" />
log:

performing testing...

| Class Name        | Precision | Recall | F1-Score | Support |
|--------------------|------------|---------|-----------|----------|
| king penguin       | 1.00 | 0.97 | 0.99 | 1300 |
| Maltese dog        | 1.00 | 0.80 | 0.89 | 1300 |
| snow leopard       | 1.00 | 0.96 | 0.98 | 1300 |
| airliner           | 0.99 | 0.93 | 0.96 | 1300 |
| airship            | 0.99 | 0.94 | 0.96 | 1300 |
| container ship     | 1.00 | 0.91 | 0.95 | 1300 |
| soccer ball        | 1.00 | 0.86 | 0.93 | 1300 |
| sports car         | 1.00 | 0.60 | 0.75 | 1300 |
| trailer truck      | 1.00 | 0.79 | 0.88 | 1300 |
| orange             | 1.00 | 0.80 | 0.89 | 1300 |
| others             | 0.00 | 0.00 | 0.00 | 0 |
|                    |          |         |           |          |
| **Accuracy**       | — | — | **0.86** | 13000 |
| **Macro avg**      | 0.91 | 0.78 | 0.83 | 13000 |
| **Weighted avg**   | 1.00 | 0.86 | 0.92 | 13000 |


#### Random noise with epsilon = 10%
<img width="1134" height="883" alt="image" src="https://github.com/user-attachments/assets/63cb5dfd-0694-4d49-9481-85a9a8f1b341" />
log:

performing testing...

| Class Name        | Precision | Recall | F1-Score | Support |
|--------------------|------------|---------|-----------|----------|
| king penguin       | 1.00 | 0.94 | 0.97 | 1300 |
| Maltese dog        | 1.00 | 0.77 | 0.87 | 1300 |
| snow leopard       | 1.00 | 0.94 | 0.97 | 1300 |
| airliner           | 0.99 | 0.89 | 0.94 | 1300 |
| airship            | 0.98 | 0.91 | 0.95 | 1300 |
| container ship     | 1.00 | 0.82 | 0.90 | 1300 |
| soccer ball        | 1.00 | 0.81 | 0.89 | 1300 |
| sports car         | 1.00 | 0.55 | 0.71 | 1300 |
| trailer truck      | 0.99 | 0.73 | 0.84 | 1300 |
| orange             | 1.00 | 0.75 | 0.86 | 1300 |
| others             | 0.00 | 0.00 | 0.00 | 0 |
|                    |          |         |           |          |
| **Accuracy**       | — | — | **0.81** | 13000 |
| **Macro avg**      | 0.91 | 0.74 | 0.81 | 13000 |
| **Weighted avg**   | 1.00 | 0.81 | 0.89 | 13000 |


## FGSM attacks
### Batch-wise FGSM
<img width="1123" height="882" alt="image" src="https://github.com/user-attachments/assets/31f3c397-1822-456e-ac40-95c2182bc333" />
log:

performing testing...

| Class Name        | Precision | Recall | F1-Score | Support |
|--------------------|------------|---------|-----------|----------|
| king penguin       | 1.00 | 0.83 | 0.91 | 1300 |
| Maltese dog        | 1.00 | 0.84 | 0.91 | 1300 |
| snow leopard       | 1.00 | 0.81 | 0.89 | 1300 |
| airliner           | 0.99 | 0.91 | 0.95 | 1300 |
| airship            | 0.98 | 0.91 | 0.95 | 1300 |
| container ship     | 1.00 | 0.71 | 0.83 | 1300 |
| soccer ball        | 1.00 | 0.64 | 0.78 | 1300 |
| sports car         | 1.00 | 0.55 | 0.71 | 1300 |
| trailer truck      | 1.00 | 0.76 | 0.86 | 1300 |
| orange             | 1.00 | 0.72 | 0.83 | 1300 |
| others             | 0.00 | 0.00 | 0.00 | 0 |
|                    |          |         |           |          |
| **Accuracy**       | — | — | **0.77** | 13000 |
| **Macro avg**      | 0.91 | 0.70 | 0.78 | 13000 |
| **Weighted avg**   | 1.00 | 0.77 | 0.86 | 13000 |


#### Sample-wise FGSM
<img width="1136" height="881" alt="image" src="https://github.com/user-attachments/assets/dfbe7ee2-abb0-4bf8-8eac-8b3a6abf6acc" />
log:

performing testing...

| Class Name        | Precision | Recall | F1-Score | Support |
|--------------------|------------|---------|-----------|----------|
| king penguin       | 1.00 | 0.84 | 0.91 | 1300 |
| Maltese dog        | 1.00 | 0.85 | 0.92 | 1300 |
| snow leopard       | 1.00 | 0.81 | 0.90 | 1300 |
| airliner           | 0.99 | 0.90 | 0.94 | 1300 |
| airship            | 0.99 | 0.91 | 0.95 | 1300 |
| container ship     | 1.00 | 0.71 | 0.83 | 1300 |
| soccer ball        | 1.00 | 0.65 | 0.78 | 1300 |
| sports car         | 1.00 | 0.56 | 0.71 | 1300 |
| trailer truck      | 1.00 | 0.75 | 0.86 | 1300 |
| orange             | 1.00 | 0.74 | 0.85 | 1300 |
| others             | 0.00 | 0.00 | 0.00 | 0 |
|                    |          |         |           |          |
| **Accuracy**       | — | — | **0.77** | 13000 |
| **Macro avg**      | 0.91 | 0.70 | 0.79 | 13000 |
| **Weighted avg**   | 1.00 | 0.77 | 0.87 | 13000 |



