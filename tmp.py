import clip
import torch
import torch.nn as nn
import csv
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from autoattack import AutoAttack
from torchvision.transforms import ToPILImage

def _convert_image_to_rgb(image):
    # Assuming the function converts non-RGB images to RGB. 
    # If the image is already in RGB format, it returns the image as is.
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


class ModelWrapper(nn.Module):
    def __init__(self, classifier, clip_model, preprocess, resolution):
        super(ModelWrapper, self).__init__()
        self.classifier = classifier
        self.clip_model = clip_model
        self.preprocess = preprocess

    def forward(self, images):
        to_pil = ToPILImage()
        pil_images = [to_pil(img) for img in images]
        images = [preprocess(img) for img in pil_images]
        images = torch.stack(images).to(device)  # Convert the list of tensors to a single tensor
        features = self.clip_model.encode_image(images)
        outputs = self.classifier(features.float().to(device))
        return outputs



# Load the model and preprocessing 
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('RN50', device)
batch_size = 128

print("load model")
classifier = torch.load("/data/gpfs/projects/punim2103/classifier_model_full.pth", map_location=device)
resolution = 224  # specify the input resolution for your CLIP model
wrapped_model = ModelWrapper(classifier, model, preprocess, resolution).to(device)
wrapped_model.eval()

# load data
print("load data")
test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transforms.ToTensor())
test_subset = Subset(test_dataset, range(1000))
test_loader = DataLoader(test_subset, batch_size=batch_size)


# define the attacker
#epsilon = 0.001
#adversary = AutoAttack(wrapped_model, norm='Linf', eps=epsilon, version='custom',device= device, attacks_to_run=['apgd-ce'])
#adversary = AutoAttack(wrapped_model, norm='Linf', eps=epsilon, version='standard',device= device)

import torch


# Initial setup
total_correct = 0  # Total number of correct predictions
total_images = 0  # Total number of images processed
batch = 0

for images, labels in test_loader:
    print("start attack")
    batch += 1
    print("batch "+str(batch))
    
    # Move images and labels to the appropriate device (GPU/CPU)
    images, labels = images.to(device), labels.to(device)

    outputs = wrapped_model(images)
    _, predicted = torch.max(outputs, 1)
    initial_acc = (predicted == labels).sum().item()
    
    # Aggregate the results
    total_correct += initial_acc
    total_images += images.size(0)  # Add the number of images in the current batch to the total

    print('Initial Accuracy for Batch {}: {:.2f}%'.format(batch, 100 * initial_acc / images.size(0)))    

    print("done")
    if total_images >= 1000:  # Stop after processing 1000 images
        break

# Compute and print the total accuracy
total_accuracy = 100 * total_correct / total_images
print('Total Accuracy Over All Images: {:.2f}%'.format(total_accuracy))
