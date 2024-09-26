import torch
from torchvision import models, transforms
from PIL import Image
import os

def load_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def segment_image(image_tensor):
    model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)
    return predictions

def process_single_image(image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if image_path.endswith(('.png', '.jpg', '.jpeg')):
        image_tensor = load_image(image_path)
        predictions = segment_image(image_tensor)

        # Save the predictions (segmentation masks and bounding boxes)
        output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_segmentation.pt")
        torch.save(predictions, output_file)
        print(f"Saved segmentation results to {output_file}")
    else:
        print("Unsupported file format. Please use .png, .jpg, or .jpeg files.")

if __name__ == "__main__":
    image_path = 'Data/input_images/11.jpg'  # Replace with your single image path
    output_dir = 'Data/segmentation_output/'
    process_single_image(image_path, output_dir)
