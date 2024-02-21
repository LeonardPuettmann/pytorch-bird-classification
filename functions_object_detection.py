import sys
import math
import os
from PIL import Image
from pathlib import Path

import imgaug as ia
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import torch
from torchvision.utils import draw_bounding_boxes, make_grid
from torchvision.io import read_image
import torchvision.transforms.functional as F
import torchvision.transforms as T


def display_image_with_boxes(image_path, bounding_boxes):
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
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    loss_values = []
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Pass targets to model only in training mode
        model.train()
        loss_dict = model(images, targets)

        # Check if loss_dict is a dictionary
        if not isinstance(loss_dict, dict):
            print(f"Unexpected type for loss_dict at batch {i}")
            print(f"Skipping this batch")
            continue

        loss_classifier = loss_dict["loss_classifier"]
        loss_box_reg = loss_dict["loss_box_reg"]
        loss_objectness = loss_dict["loss_objectness"]
        loss_rpn_box_reg = loss_dict["loss_rpn_box_reg"]

        losses = sum([loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg])
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_values.append(loss_value)

        # # Get the model's predictions
        with torch.no_grad():
            model.eval()
            predictions = model(images)

            labels = [p["labels"].item() if p["labels"].nelement() == 1 else 0 for p in predictions]
            class_targets = [t["labels"].item() for t in targets]

            # Calculate the number of correct predictions
            correct = sum(l == t for l, t in zip(labels, class_targets))

            # Calculate the accuracy
            accuracy = correct / len(labels)

        print(f"Batch {i} of {len(data_loader)} | 'accuracy': {accuracy} | 'loss overall': {loss_value} | 'loss classifier' {loss_classifier}")

    metrics = {'loss_values': loss_values}
    return metrics


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        if isinstance(img, torch.Tensor):
            img = img.detach()
            img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def display_image_with_boxes(model, image_path, transform, color="red"):
    # resize image using the Image library 
    image = Image.open(image_path)

    # transform the image for the model 
    image_transformed = transform(image)
    image_transformed = image_transformed.unsqueeze(0)

    # transform the image for viz
    uint_image = read_image(image_path)
    print(uint_image.shape)

    # Delete the temporary image file
    #os.remove(temp_path)    

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
    labels = [class_names[label-2]]

    font_path = "c:\Windows\Fonts\CONSOLA.TTF"
    result = draw_bounding_boxes(uint_image, boxes=boxes, colors=[color], labels=labels, width=12, font=font_path, font_size=90)

    # Convert the tensor image to PIL image for displaying
    result = F.to_pil_image(result)

    show(result)