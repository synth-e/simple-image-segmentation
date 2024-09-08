import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import json
from PIL import Image

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


class SegmentationDataset(Dataset):
    def __init__(
        self,
        root_dir,
        image_dir="images",
        mask_dir="masks",
        transform=None,
        target_transform=None,
    ):
        """
        Args:
            root_dir (str): Directory with all the images and masks.
            image_dir (str): Subdirectory name containing the input images.
            mask_dir (str): Subdirectory name containing the segmentation masks.
            transform (callable, optional): Optional transform to be applied on an image.
            target_transform (callable, optional): Optional transform to be applied on a mask.
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, image_dir)
        self.mask_dir = os.path.join(root_dir, mask_dir)
        self.transform = transform
        self.target_transform = target_transform

        # Get the list of image and mask filenames
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.mask_filenames = sorted(os.listdir(self.mask_dir))

        # Ensure that the number of images and masks are the same
        assert len(self.image_filenames) == len(
            self.mask_filenames
        ), "Mismatch between images and masks."

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load the image and mask
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        assert (
            self.image_filenames[idx].split(".")[0]
            == self.mask_filenames[idx].split(".")[0]
        )

        image = Image.open(img_path).convert("RGB")  # Convert image to RGB
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        # Apply transformations if they exist
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask


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

    # Load Dataset
    train_dataset = SegmentationDataset(
        root_dir=args.data,
        transform=input_transform,
        target_transform=target_transform,
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )

    # Initialize Model
    model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
    model.classifier[4] = nn.Conv2d(
        256, args.num_classes, kernel_size=(1, 1), stride=(1, 1)
    )  # Adjust for number of classes
    model = model.cuda()  # Move model to GPU if available

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

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
    # parser.add_argument(
    #     "--label",
    #     type=str,
    #     required=True,
    #     help="Path to the label mapping file, expect json file",
    # )
    parser.add_argument(
        "--output", type=str, required=True, help="File path to save the trained model"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        required=True,
        help="Number of objects",
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
