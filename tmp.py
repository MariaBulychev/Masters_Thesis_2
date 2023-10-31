import clip
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from autoattack import AutoAttack
from torchvision.transforms import ToPILImage

def _convert_image_to_rgb(image):
    if image.mode != "RGB":
        return image.convert("RGB")
    return image

class ModelWrapper(nn.Module):
    def __init__(self, classifier, clip_model, resolution):
        super(ModelWrapper, self).__init__()
        self.classifier = classifier
        self.clip_model = clip_model
        self.preprocess = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def forward(self, images):
        features = self.clip_model.encode_image(images)
        outputs = self.classifier(features.float().to(device))
        return outputs

# Load the model and preprocessing 
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('RN50', device)
batch_size = 16

print("load model")
classifier = torch.load("/data/gpfs/projects/punim2103/classifier_model_full.pth", map_location=device)
resolution = 224
wrapped_model = ModelWrapper(classifier, model, resolution).to(device)
wrapped_model.eval()

# Load data
print("load data")
test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transforms.ToTensor())
test_subset = Subset(test_dataset, range(1000))
test_loader = DataLoader(test_subset, batch_size=batch_size)

# Initial setup
total_correct = 0
total_images = 0
batch = 0

to_pil = transforms.ToPILImage()  # Define ToPILImage transform here
preprocess = wrapped_model.preprocess  # Get preprocessing pipeline from the model wrapper

for images, labels in test_loader:
    print("start attack")
    batch += 1
    print("batch "+str(batch))
    images, labels = images.to(device), labels.to(device)

    # Convert images to PIL format and preprocess
    images = torch.stack([preprocess(to_pil(img)) for img in images])

    outputs = wrapped_model(images)
    _, predicted = torch.max(outputs, 1)
    initial_acc = (predicted == labels).sum().item()

    total_correct += initial_acc
    total_images += images.size(0)

    print('Initial Accuracy for Batch {}: {:.2f}%'.format(batch, 100 * initial_acc / images.size(0)))

    print("done")
    if total_images >= 1000:
        break

total_accuracy = 100 * total_correct / total_images
print('Total Accuracy Over All Images: {:.2f}%'.format(total_accuracy))
