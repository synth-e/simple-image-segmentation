import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
import argparse
import os
import json

# Define Data Transformations
input_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

target_transform = transforms.Compose(
    [transforms.Resize((256, 256)), transforms.ToTensor()]
)


# Custom Dataset Class
class PascalVOCDataset(VOCSegmentation):
    def __init__(
        self,
        root,
        label_mapping,
        image_set="train",
        transform=None,
        target_transform=None,
    ):
        super().__init__(
            root=root,
            year="2012",
            image_set=image_set,
            download=False,
            transform=transform,
            target_transform=target_transform,
        )
        self.label_mapping = label_mapping

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        target = self._map_labels(target)
        return image, target

    def _map_labels(self, mask):
        # Map the labels in the mask based on the label_mapping
        mapped_mask = torch.zeros_like(mask, dtype=torch.long)
        for key, value in self.label_mapping.items():
            mapped_mask[mask == int(key)] = value
        return mapped_mask


# Define Training Loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.cuda(), masks.long().cuda()

            # Forward pass
            outputs = model(images)["out"]
            loss = criterion(outputs, masks.squeeze(1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}"
        )

    print("Finished Training")


# Load label mapping from the file
def load_label_mapping(label_file):
    with open(label_file, "r") as f:
        label_mapping = json.load(f)
    # Map values to indices for masks (e.g., {0: background, 1: item1, 2: item2})
    return {int(k): int(v) for k, v in label_mapping.items()}


# Main Function
def main(args):
    # Check if dataset path exists
    if not os.path.isdir(args.data):
        raise ValueError(f"Dataset path {args.data} does not exist.")

    # Load the label mapping from the provided JSON file
    if not os.path.isfile(args.label):
        raise ValueError(f"Label file {args.label} does not exist.")
    label_mapping = load_label_mapping(args.label)

    # Load Dataset
    train_dataset = PascalVOCDataset(
        root=args.data,
        label_mapping=label_mapping,
        image_set="train",
        transform=input_transform,
        target_transform=target_transform,
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )

    # Initialize Model
    model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
    model.classifier[4] = nn.Conv2d(
        256, len(label_mapping), kernel_size=(1, 1), stride=(1, 1)
    )  # Adjust for number of classes
    model = model.cuda()  # Move model to GPU if available

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train Model
    train_model(model, train_loader, criterion, optimizer, num_epochs=args.num_epochs)

    # Save Trained Model
    torch.save(model.state_dict(), args.output)
    print(f"Model saved at {args.output}")


if __name__ == "__main__":
    # Argument Parser for command-line execution
    parser = argparse.ArgumentParser(
        description="Train a segmentation model on Pascal VOC dataset."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the dataset directory (contains images and masks)",
    )
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Path to the label mapping file, expect json file",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="File path to save the trained model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        required=False,
        help="Batch size to train model",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        required=False,
        help="Number of epochs to train model",
    )

    args = parser.parse_args()

    # Call the main function
    main(args)
