# Lab1: test Noise and FGSM- attacks on imagenet-pretrained architecture using ImageNet-10 dataset.
https://www.kaggle.com/datasets/liusha249/imagenet10

The architecture chosen was MobileNetV3-Large. the model outputs 1000 classes, so the mapping was used - new "others" class is added to the 10 available in dataset. if the class predicted is not one of the ten selected - its automatically passed as "others" into prediction data and confusion matrices.
