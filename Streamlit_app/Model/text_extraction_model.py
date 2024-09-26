import os
import easyocr
import json

def extract_text(image_path):
    try:
        reader = easyocr.Reader(['en'])
        result = reader.readtext(image_path)

        # Extract only the text from the results
        text = ' '.join([item[1] for item in result if item[1]])
        
        # Return the text in the desired format
        return text if text else "No text available"
    except Exception as e:
        print(f"Error during text extraction: {e}")
        return "Error during extraction"

def save_results_to_file(results, output_file):
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"Error saving results to file: {e}")

if __name__ == "__main__":
    image_path = 'Data/segmented_objects/11_object_0.png'
    output_file = 'Data/extracted_text.json'
    
    # Extract text from the single image
    text_data = extract_text(image_path)
    
    # Save the results to a file
    save_results_to_file({"text": text_data}, output_file)
    
    print(f"Text data extracted and saved to {output_file}")
