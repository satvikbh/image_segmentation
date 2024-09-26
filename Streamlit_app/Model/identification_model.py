import os
import json
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

def identify_object(image_path):
    # Load the pre-trained CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load and preprocess the image
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")

    # Perform forward pass through the model to get image features
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    # Use a placeholder list of possible object labels
    object_labels = ["dog", "cat", "car", "tree", "bicycle", "person", "chair", "cup", "book", "phone", "road sign", "Football","bird","building"]
    
    # Tokenize the text labels
    text_inputs = processor(text=object_labels, return_tensors="pt", padding=True)
    
    # Perform forward pass to get text features
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)

    # Compute cosine similarity between image features and text features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    similarities = torch.matmul(image_features, text_features.T)

    # Identify the best match
    best_match_idx = similarities.argmax().item()
    best_match_label = object_labels[best_match_idx]

    # Prepare the description output (excluding confidence score)
    description = {
        "description": f"Identified as: {best_match_label}"
    }

    return description

def save_description_to_file(description, output_file):
    with open(output_file, 'w') as f:
        json.dump(description, f, indent=4)

if __name__ == "__main__":
    # Input single image file
    image_path = 'Data/segmented_objects/11_object_0.png'
    image_filename = os.path.basename(image_path)  # Extract the filename

    # Get the description
    description = identify_object(image_path)

    # Prepare description data in the desired format
    description_dict = {
        image_filename: description
    }

    # Save description to file
    output_file = 'Data/object_description.json'
    save_description_to_file(description_dict, output_file)
    print(f"Description saved to {output_file}")
