import torch
from torchvision import transforms
import os
import json
from PIL import Image

def load_segmentation(segmentation_path):
    return torch.load(segmentation_path)

def extract_and_save_objects(predictions, image_path, output_dir, image_name):
    image_tensor = transforms.ToTensor()(Image.open(image_path)).unsqueeze(0)
    os.makedirs(output_dir, exist_ok=True)
    image_id = image_name.split('.')[0]
    master_id = f"{image_id}_master"

    for i, mask in enumerate(predictions[0]['masks']):
        mask = mask[0]  # Assuming mask is in the correct format
        object_image = image_tensor * mask
        object_image_pil = transforms.ToPILImage()(object_image.squeeze(0))
        object_id = f"{image_id}_object_{i}"
        object_image_pil.save(f"{output_dir}/{object_id}.png")

        # Save metadata
        metadata = {
            "object_id": object_id,
            "master_id": master_id,
            "image_name": image_name,
            "index": i,
        }
        metadata_path = os.path.join(output_dir, f"{object_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

def process_single_image(image_path, segmentation_path, output_dir):
    if os.path.exists(segmentation_path):
        predictions = load_segmentation(segmentation_path)
        image_name = os.path.basename(image_path)
        extract_and_save_objects(predictions, image_path, output_dir, os.path.splitext(image_name)[0])
    else:
        print(f"Segmentation file {segmentation_path} not found.")

if __name__ == "__main__":
    image_path = 'Data/input_images/11.jpg'  # Replace with your single image path
    segmentation_path = 'Data/segmentation_output/11_segmentation.pt'  # Replace with corresponding segmentation file path
    output_dir = 'Data/segmented_objects/'
    process_single_image(image_path, segmentation_path, output_dir)
