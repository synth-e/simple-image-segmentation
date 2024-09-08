import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os

# Define transformation for input image (resize, convert to tensor, and normalize)
input_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_model(model_path, num_classes):
    """
    Loads a trained segmentation model from a .pth file.
    """
    # Load the DeepLabV3 model
    model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(
        256, num_classes, kernel_size=(1, 1), stride=(1, 1)
    )  # Adjust for number of classes

    # Load the trained weights
    model.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode
    model.eval()

    return model


def predict(model, image_path):
    """
    Performs prediction on a single input image.
    """
    # Open the image
    image = Image.open(image_path).convert("RGB")

    # Preprocess the image
    input_image = input_transform(image).unsqueeze(0)  # Add batch dimension

    # Move image to the same device as the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_image = input_image.to(device)

    # Perform prediction
    with torch.no_grad():
        output = model(input_image)["out"][0]

    # Get the predicted class for each pixel
    predicted_mask = torch.argmax(output, dim=0).cpu().numpy()

    return predicted_mask, image


def visualize_segmentation(image, mask):
    """
    Visualizes the input image and its predicted segmentation mask.
    """
    plt.figure(figsize=(10, 5))

    # Show input image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")

    # Show predicted mask
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="jet")
    plt.title("Predicted Segmentation Mask")
    
    try:
        plt.show()
    except:
        plt.savefig('out.png')


def main(args):
    # Load the trained model
    model = load_model(args.model, args.num_classes)

    # Perform prediction on the input image
    predicted_mask, image = predict(model, args.image)

    # Visualize the result
    visualize_segmentation(image, predicted_mask)


if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate a segmentation model on a single image."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the trained .pth model file."
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        required=True,
        help="Number of objects",
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the input image."
    )

    args = parser.parse_args()

    # Run the evaluation
    main(args)
