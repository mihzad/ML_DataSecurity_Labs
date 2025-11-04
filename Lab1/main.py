


import math
import numpy as np

import torch
from torchvision.transforms import v2
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from noise_uniform import UniformNoise
from fgsm import batch_fgsm_attack, single_fgsm_attack

from PIL import Image
import matplotlib.pyplot as plt


img_size = 224
rand_state = 44
num_workers = 8
device = torch.device("cuda")

def custom_loader(path):
    return Image.open(path, formats=["JPEG"]).convert("RGB")


def printshare(msg, logfile="testing_log.txt"):
    print(msg)

    with open(logfile, "a") as f:
        print(msg, file=f)


imagenet_labels = [ #ordered as in ImageFolder
    "king penguin",
    "Maltese dog",
    "snow leopard",
    "airliner",
    "airship",
    "container ship",
    "soccer ball",
    "sports car",
    "trailer truck",
    "orange",
]

imagenet_code_to_label = { #ordered as in ImageFolder
    "n02056570": "king penguin",
    "n02085936": "Maltese dog",
    "n02128757": "snow leopard",
    "n02690373": "airliner",
    "n02692877": "airship",
    "n03095699": "container ship",
    "n04254680": "soccer ball",
    "n04285008": "sports car",
    "n04467665": "trailer truck",
    "n07747607": "orange",
}

# Load ImageNet class index mapping from torchvision
imagenet_classes_all = MobileNet_V3_Large_Weights.IMAGENET1K_V2.meta["categories"]

# Build a map from class index → reduced index
class_map = {}
for i, name in enumerate(imagenet_classes_all):
    if name in imagenet_labels:
        class_map[i] = imagenet_labels.index(name)

others_idx = len(imagenet_labels)  # index for "others"


def uniform_noise_fn(amount: float = 0.05):
    """Return a callable that adds uniform noise in [-amount, +amount]."""
    def fn(x: torch.Tensor):
        # x shape [C,H,W], values in [0,1]
        noise = (torch.rand_like(x).sub(0.5)).mul(2.0 * amount)
        out = (x + noise).clamp(0.0, 1.0)
        return out
    return fn

def remap_indices(indices):
    new_indices = []
    for i in indices:
        if int(i) in class_map:
            new_indices.append(class_map[int(i)])
        else:
            new_indices.append(others_idx)
    return np.array(new_indices)

def perform_test(net, testing_set, batch_size, saveto:str):
    printshare("performing testing...")
    testing_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    net.eval()

    correct = 0
    total = 0
    targets = []
    predictions = []
    for inputs, labels in testing_loader:
        inputs, labels = inputs.cuda(), labels.cuda()

        #adv_inputs = batch_fgsm_attack(net, inputs, labels, epsilon=8/255.0)
        adv_inputs = []
        for i in range(len(inputs)):
            adv_inputs.append(single_fgsm_attack(net, inputs[i], labels[i], epsilon=8/255.0))
        adv_inputs = torch.stack(adv_inputs)
        with torch.no_grad():
            outputs = net(adv_inputs)

            pred_vals, pred_classes = torch.max(outputs.data, 1)
            correct += (pred_classes == labels).sum().item()
            total += labels.size(0)
            preds_corrected = remap_indices(pred_classes.detach().cpu().numpy())
            targets.extend(labels.detach().cpu().numpy())
            predictions.extend(preds_corrected)

    torch.save({  # stats
        'targets': targets,
        'predictions': predictions,
    },
        saveto)

    return targets, predictions

def run_confusion_matrix(dataset, targets, predictions):
    cm = confusion_matrix(y_true=targets, y_pred=predictions, normalize="true")
    cm = np.round(cm, 3)

    display_labels = [imagenet_code_to_label[k] for k in dataset.classes] + ["others"]

    cmp = ConfusionMatrixDisplay(cm, display_labels=display_labels)

    ax = plt.subplot()
    plt.rcParams.update({'font.size': 6})
    label_font = {'size': '13'}
    ax.set_xlabel('Predicted labels', fontdict=label_font)
    ax.set_ylabel('Observed labels', fontdict=label_font)
    title_font = {'size': '16'}
    ax.set_title('Confusion Matrix', fontdict=title_font)
    cmp.plot(ax=ax)

    plt.show()

    printshare(classification_report(y_true=targets, y_pred=predictions, target_names=display_labels))


if __name__ == '__main__':
    net = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    net.to(device)

    heavy_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.75, 1.0),
            ratio=(7.0 / 8.0, 8.0 / 7.0)
        ),
        v2.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),

        # edge padding before + centercrop after rotation => corners filled.
        v2.Pad(padding_mode="edge", padding=math.ceil(img_size * 0.2)),
        v2.RandomRotation(
            degrees=15,
            interpolation=v2.InterpolationMode.BILINEAR
        ),
        v2.CenterCrop(img_size),

        v2.RandomHorizontalFlip(p=0.5),
        v2.GaussianNoise(mean=0, sigma=0.08),
        v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )

    ])

    light_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.75, 1.0),
            ratio=(7.0 / 8.0, 8.0 / 7.0)
        ),
        v2.RandomHorizontalFlip(p=0.5),

        v2.ToDtype(torch.float32, scale=True)
        #UniformNoise(amount=0.1),
        #v2.Normalize(
        #    mean=(0.485, 0.456, 0.406),
        #    std=(0.229, 0.224, 0.225)
        #)
    ])
    checkfile = "mobilenetv3-fgsm_singlewise_attack-predictiondata.pth"
    dataset = datasets.ImageFolder('imagenet-10', transform=light_transform, loader=custom_loader)
    targets, preds = perform_test(net, dataset, batch_size=64, saveto=checkfile)
    #check = torch.load(checkfile, weights_only=False)
    #targets, preds = check['targets'], check['predictions']
    run_confusion_matrix(dataset, targets, preds)

