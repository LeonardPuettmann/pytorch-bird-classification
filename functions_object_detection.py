import sys
import math
import os
from PIL import Image
from pathlib import Path
from typing import List

import imgaug as ia
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import numpy as np

import torch
import torchvision
from torchvision.utils import draw_bounding_boxes, make_grid
from torchvision.io import read_image
import torchvision.transforms.functional as F
import torchvision.transforms as T

import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils

# set device so that it does not need to be set as an argument
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def display_image_with_boxes(image_path: str, bounding_boxes: List):
    """
    Display an image with bounding boxes.

    Parameters:
    - image_path: str, path to the image file
    - bounding_boxes: list of lists, each inner list contains four values (x, y, width, height)
    """

    # Open the image file
    img = Image.open(image_path)
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(img)

    # Create a Rectangle patch for each bounding box and add it to the plot
    for box in bounding_boxes:
        x, y, width, height = box
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Show the plot
    plt.show()


def collate_fn(batch):
    """
    Simple function to convert a batch into a tuple.
    """
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def show(imgs):
    """
    Helper function to display an image.
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        if isinstance(img, torch.Tensor):
            img = img.detach()
            img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def display_prediction_with_boxes(model: torch.nn.Module, image_path: str, transform: torchvision.transforms, color: str ="red"):
    """
    Heplper function to get predictions from a Faster RCNN model and print these predictions on an image. 
    This model is for an object detection task, so it will vizualize the bounding boxes for a found object, 
    in this case the the bounding boxes should be for a bird.

    Paramters: 
    - model: Faster RCNN model. 
    - image_path: The file path for the image to print and get predictions on.
    - transform: Image transformation to preprocess the image for the model. Should be a torchvision transforms.
    - color: Color of the bounding box. Red by default. 
    """

    # resize image using the Image library 
    image = Image.open(image_path)

    # transform the image for the model 
    image_transformed = transform(image)
    image_transformed = image_transformed.unsqueeze(0)

    # transform the image for viz
    uint_image = read_image(image_path)
    print(uint_image.shape)

    # Ensure the model is in evaluation mode
    model.eval()

    # Make a prediction
    with torch.no_grad():
        prediction = model(image_transformed.to(device))
        print(prediction)

    boxes = prediction[0]["boxes"]
    label = prediction[0]["labels"].item()
    print(label)
    print(boxes)

    class_names = os.listdir("cub-200-2011/images")
    labels = [class_names[label]]

    font_path = "c:\Windows\Fonts\CONSOLA.TTF"
    result = draw_bounding_boxes(uint_image, boxes=boxes, colors=[color], labels=labels, width=12, font=font_path, font_size=90)

    # Convert the tensor image to PIL image for displaying
    result = F.to_pil_image(result)

    show(result)