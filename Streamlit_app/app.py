import streamlit as st
from PIL import Image
import os
import shutil
import json
import pandas as pd

# Import functions from the Model and Utils directories
from Model.segmentation_model import process_single_image as segment_image
from Utlis.object_extraction import process_single_image as extract_objects
from Model.identification_model import identify_object, save_description_to_file
from Model.text_extraction_model import extract_text, save_results_to_file as save_text
from Model.summarization_model import summarize_attributes, save_results_to_file as save_summaries
from Utlis.data_mapping import map_data, save_mapped_data
from Utlis.visualization import annotate_image, generate_summary_table, load_mapped_data

# Paths for temporary files and outputs
output_dir = 'Data/'
segmentation_output_dir = os.path.join(output_dir, 'segmentation_output')
objects_output_dir = os.path.join(output_dir, 'segmented_objects')
final_output_dir = os.path.join(output_dir, 'FINAL_OUTPUT')
os.makedirs(output_dir, exist_ok=True)

# Function to clear previous data
def clear_previous_data():
    if os.path.exists(segmentation_output_dir):
        shutil.rmtree(segmentation_output_dir)
    if os.path.exists(objects_output_dir):
        shutil.rmtree(objects_output_dir)
    if os.path.exists(final_output_dir):
        shutil.rmtree(final_output_dir)

    os.makedirs(segmentation_output_dir, exist_ok=True)
    os.makedirs(objects_output_dir, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)

# Streamlit UI
st.title("Image Processing Pipeline")

# File Upload
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    clear_previous_data()

    # Process the new image
    image = Image.open(uploaded_file)
    image_path = os.path.join(output_dir, uploaded_file.name)
    image.save(image_path)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Step 1: Segmentation
    st.subheader("1. Segmentation")
    segment_image(image_path, segmentation_output_dir)
    st.write(f"Segmentation completed. Results saved in {segmentation_output_dir}.")

    # Step 2: Object Extraction
    st.subheader("2. Object Extraction")
    segmentation_file = os.path.join(segmentation_output_dir, f"{os.path.splitext(uploaded_file.name)[0]}_segmentation.pt")
    extract_objects(image_path, segmentation_file, objects_output_dir)
    st.write(f"Objects extracted and saved in {objects_output_dir}.")

    # Display extracted objects
    st.subheader("Extracted Objects")
    object_images = [f for f in os.listdir(objects_output_dir) if f.endswith('.png')]
    for obj_img in object_images:
        st.image(os.path.join(objects_output_dir, obj_img), caption=obj_img, use_column_width=True)

    # Step 3: Object Identification
    st.subheader("3. Object Identification")
    description_output_file = os.path.join(output_dir, 'object_description.json')
    descriptions = {}

    for obj_img in object_images:
        obj_path = os.path.join(objects_output_dir, obj_img)
        description = identify_object(obj_path)
        
        descriptions[obj_img] = {
            "description": description.get('description', 'No description available.'),
            "text": None
        }

    save_description_to_file(descriptions, description_output_file)
    st.write(f"Object descriptions saved in {description_output_file}.")

    # Step 4: Text Extraction
    st.subheader("4. Text Extraction")
    text_output_file = os.path.join(output_dir, 'extracted_text.json')
    texts = {}
    for obj_img in object_images:
        obj_path = os.path.join(objects_output_dir, obj_img)
        extracted_text = extract_text(obj_path)

        texts[obj_img] = extracted_text
        descriptions[obj_img]['text'] = extracted_text

    save_text(texts, text_output_file)
    st.write(f"Extracted text saved in {text_output_file}.")

    # Step 5: Summarization
    st.subheader("5. Summarization")
    summaries_output_file = os.path.join(output_dir, 'summaries.json')
    summaries = summarize_attributes(texts)
    save_summaries(summaries, summaries_output_file)
    st.write(f"Summaries saved in {summaries_output_file}.")

    # Step 6: Data Mapping
    st.subheader("6. Data Mapping")
    mapped_data_output_file = os.path.join(output_dir, 'mapped_data.json')
    mapped_data = map_data(descriptions, texts, summaries)
    save_mapped_data(mapped_data, mapped_data_output_file)
    st.write(f"Mapped data saved in {mapped_data_output_file}.")

    # Step 7: Visualization
    st.subheader("7. Visualization")
    annotated_image_path = os.path.join(final_output_dir, f"{os.path.splitext(uploaded_file.name)[0]}_annotated.jpg")
    summary_table_path = os.path.join(final_output_dir, 'summary_table.txt')

    annotate_image(image_path, mapped_data, annotated_image_path, add_text=False)
    generate_summary_table(mapped_data, summary_table_path)

    st.image(annotated_image_path, caption="Annotated Image", use_column_width=True)

    # Display the summary table in a tabular format
    st.subheader("Summary Table")

    try:
        summary_df = pd.read_csv(summary_table_path, sep='\t')
        if summary_df.empty:
            st.write("No summary available.")
        else:
            st.dataframe(summary_df)
    except Exception as e:
        st.write(f"Error loading summary table: {e}")
