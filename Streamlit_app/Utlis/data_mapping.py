import json

def load_json_data(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not a valid JSON file.")
        return {}

def map_data(descriptions, texts, summaries):
    mapped_data = {}
    
    for obj_id in descriptions.keys():
        # Format the description field correctly
        mapped_data[obj_id] = {
            'description': {
                'description': descriptions.get(obj_id, {}).get('description', 'No description available')
            },
            'text': texts.get(obj_id, 'No text extracted'),
            'summary': summaries.get(obj_id, 'No summary available')
        }
    
    return mapped_data

def save_mapped_data(mapped_data, output_file):
    with open(output_file, 'w') as f:
        json.dump(mapped_data, f, indent=4)

if __name__ == "__main__":
    descriptions_file = 'Data/object_description.json'
    texts_file = 'Data/extracted_text.json'
    summaries_file = 'Data/summaries.json'
    output_file = 'Data/mapped_data.json'

    # Load descriptions, texts, and summaries
    descriptions = load_json_data(descriptions_file)
    texts = load_json_data(texts_file)
    summaries = load_json_data(summaries_file)
    
    # Map the data to include only descriptions in the desired format
    mapped_data = map_data(descriptions, texts, summaries)
    
    # Save the mapped data to a JSON file
    save_mapped_data(mapped_data, output_file)
    
    print(f"Mapped data saved to {output_file}")
