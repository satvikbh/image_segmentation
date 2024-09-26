import json
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

def load_mapped_data(mapped_data_file):
    try:
        with open(mapped_data_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {mapped_data_file} does not exist.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: The file {mapped_data_file} is not a valid JSON file.")
        return {}

def annotate_image(image_path, mapped_data, output_image_path, add_text=True):
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_image_path)
    os.makedirs(output_dir, exist_ok=True)
    
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    # Load a font (adjust the path to the font as needed)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    
    if add_text:
        for obj_id, data in mapped_data.items():
            description = data.get('description', 'No description available.')
            text = data.get('text', 'No text available.')
            summary = data.get('summary', 'No summary available.')
            
            # Draw the description, text, and summary on the image
            annotation_text = f"ID: {obj_id}\nDescription: {description}\nText: {text}\nSummary: {summary}"
            
            # For simplicity, place text at a fixed position
            draw.text((10, 10 + 20 * int(obj_id)), annotation_text, fill="white", font=font)
    
    # Save the annotated image
    image.save(output_image_path)
    print(f"Annotated image saved to {output_image_path}")

def generate_summary_table(mapped_data, output_table_path):
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_table_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a dictionary to store summary table data
    summary_table_data = {
        "Object ID": [],
        "Description": [],
        "Text": [],
        "Summary": []
    }
    
    # Populate the summary table with data from the mapped data
    for obj_id, data in mapped_data.items():
        summary_table_data["Object ID"].append(obj_id)
        summary_table_data["Description"].append(data.get('description', 'No description available.'))
        summary_table_data["Text"].append(data.get('text', 'No text available.'))
        summary_table_data["Summary"].append(data.get('summary', 'No summary available.'))
    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(summary_table_data)
    
    # Save the DataFrame to a text file (tab-separated)
    df.to_csv(output_table_path, sep='\t', index=False)
    print(f"Summary table saved to {output_table_path}")

if __name__ == "__main__":
    # Paths to input files
    mapped_data_file = 'Data/mapped_data.json'
    image_path = 'Data/input_images/11.jpg'  # Use the original image path
    
    # Paths to output files
    output_image_path = 'Data/FINAL_OUTPUT/11_annotated.jpg'
    output_table_path = 'Data/FINAL_OUTPUT/summary_table.txt'
    
    # Load the mapped data
    mapped_data = load_mapped_data(mapped_data_file)
    
    # Annotate the image with the processed data
    annotate_image(image_path, mapped_data, output_image_path, add_text=False)
    
    # Generate a summary table
    generate_summary_table(mapped_data, output_table_path)
