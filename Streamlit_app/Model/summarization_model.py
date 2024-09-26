from transformers import pipeline
import json

def load_text_data_from_json(json_file):
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {json_file} does not exist.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: The file {json_file} is not a valid JSON file.")
        return {}

def summarize_attributes(text_data):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summaries = {}
    
    for object_id, text_content in text_data.items():
        # Check if text is available and not too short
        if not text_content or text_content == "No text available":
            summaries[object_id] = "No summary available."
        elif len(text_content.split()) <= 3:  # Adjust threshold for minimal text length
            summaries[object_id] = "No summary available."
        else:
            # Generate a summary if text is sufficiently long
            try:
                # Adjust length parameters as needed
                summary = summarizer(text_content, max_length=150, min_length=30, do_sample=False)
                summaries[object_id] = summary[0]['summary_text']
            except Exception as e:
                summaries[object_id] = f"Error generating summary: {e}"

    return summaries

def save_results_to_file(results, output_file):
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    # Path to the JSON file with extracted text data
    text_data_file = 'Data/extracted_text.json'
    
    # Path to save the summarized results
    output_file = 'Data/summaries.json'
    
    # Load the extracted text data from the JSON file
    text_data = load_text_data_from_json(text_data_file)
    
    # Generate summaries
    summaries = summarize_attributes(text_data)
    
    # Save the summaries to a file
    save_results_to_file(summaries, output_file)
    
    print(f"Summaries saved to {output_file}")
