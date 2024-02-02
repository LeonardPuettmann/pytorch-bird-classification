#!/usr/bin/python3

# Standard library imports
import copy
from datetime import datetime
import os
import random
import shutil

# Third party imports
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch


def train_one_epoch(model, criterion, optimizer, train_loader, device):
    running_loss = 0
    correct_predictions = 0
    total_predictions = 0
    loss_history = []
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loss_history.append(loss.item())

        _, predicted = torch.max(output.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        if (i + 1) % 10 == 0:
            accuracy = correct_predictions / total_predictions
            print(f'Step [{i+1}/{len(train_loader)}] | Loss: {loss.item()} | Accuracy: {accuracy}')

    return running_loss, correct_predictions, total_predictions, loss_history


def validate_one_epoch(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0
    val_correct_predictions = 0
    val_total_predictions = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            val_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            val_total_predictions += labels.size(0)
            val_correct_predictions += (predicted == labels).sum().item()

    model.train()
    return val_loss, val_correct_predictions, val_total_predictions

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, device, epochs, patience):
    best_val_loss = float('inf')
    early_stopping_counter = 0
    loss_history = []

    for epoch in range(epochs):
        running_loss, correct_predictions, total_predictions, epoch_loss_history = train_one_epoch(model, criterion, optimizer, train_loader, device)
        loss_history.extend(epoch_loss_history)

        val_loss, val_correct_predictions, val_total_predictions = validate_one_epoch(model, criterion, val_loader, device)
        val_loss /= len(val_loader)
        val_accuracy = val_correct_predictions / val_total_predictions

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss}, Accuracy: {correct_predictions / total_predictions}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            best_model = copy.deepcopy(model.state_dict())
        else:
            early_stopping_counter += 1
            print(f'EarlyStopping counter: {early_stopping_counter} out of {patience}')
            if early_stopping_counter >= patience:
                print('Early stopping')
                model.load_state_dict(best_model)
                break

        scheduler.step()

    return model, loss_history

def load_model(shell_model, model_path, device):
    state_dict = torch.load(model_path)
    shell_model.load_state_dict(state_dict)
    model = model.to(device)
    return model

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def show_random_birbs(img_dir="data\\train"):
    # Get the list of all subdirectories
    subdirs = [os.path.join(img_dir, d) for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]

    # Randomly select 8 subdirectories
    selected_dirs = random.sample(subdirs, 8)

    images = []
    labels = []

    # For each selected directory
    for dir in selected_dirs:
        # Get the list of all files in the directory
        files = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        
        # Randomly select a file
        selected_file = random.choice(files)
        
        # Add the selected file to the images list
        images.append(selected_file)
        
        # Add the name of the bird (assumed to be the name of the directory) to the labels list
        labels.append(os.path.basename(dir))

    # Create a grid of subplots
    fig, axs = plt.subplots(2, 4, figsize=(15, 8))

    # For each image and corresponding label
    for i, (image_path, label) in enumerate(zip(images, labels)):
        # Load the image
        img = mpimg.imread(image_path)
        
        # Determine the position of the subplot
        row = i // 4
        col = i % 4
        
        # Display the image in the subplot
        axs[row, col].imshow(img)
        axs[row, col].axis('off')
        
        # Display the label above the image
        axs[row, col].set_title(label)

    # Display the grid of images
    plt.show()

def move_files(from_path, also_in_path, to_path):
    # Get all subfolders in train-subset
    train_subfolders = [f.name for f in os.scandir(from_path) if f.is_dir()]

    # Iterate over each subfolder
    for subfolder in train_subfolders:
        # Check if this subfolder exists in the 'test' folder
        if os.path.isdir(os.path.join(also_in_path, subfolder)):
            # If it does, copy it over to 'test-subset'
            shutil.copytree(os.path.join(also_in_path, subfolder), os.path.join(from_path, subfolder))

def get_height_width(path):
    # Open an image file
    img = Image.open(path)
    # Get the size of the image
    width, height = img.size
    print(f'The image size is {width} x {height}')

def plot_loss(loss_history, save): 
    plt.figure(figsize=(12, 7))
    plt.plot(loss_history)
    if save:
        plt.savefig(f"./loss/loss-{datetime.now().strftime('%Y%m%d-%H%M%S')}.png")