Image Processing Pipeline
Overview
This project is developed for image segmentation and to analyze , identify and store the objects within a image and display the outputs in a summary table by mapping the data such as object id , object discription,text present , summary of extracted text in a table format 
I perform image segmentation in majorly  7 steps such as:-
1. image segmentation 
2. object extraction 
3. object identification 
4. text extraction
5. summerization 
6. Data Mapping 
7. Visualization of results

a streamlit UI is also develop during this project that helps the user to see or to visualize the results 
The streamlit UI has some functions such as 
1. It allows user to upload an image 
2. perform various functions such as segmentation ,Object extraction , Text extraction , summerizarion of text, data mapping an visualization 
3. It shows results to user in a table format in which it has a description column that tells about the onject category(like cat, dog , car ,person etc) and text extracted from the image and the summary of that extraacted text.


The project is structured to handle the following steps:

Segmentation: Identifies and segments objects within an image using a pre-trained Mask R-CNN model.
Object Extraction: Extracts individual objects from the segmented masks and saves them as separate images.
Object Identification: Uses CLIP model to identify and describe each extracted object.
Text Extraction: Extracts text from the segmented objects using EasyOCR.
Summarization: Summarizes the extracted text using a BART model.
Data Mapping: Maps the extracted descriptions, text, and summaries into a structured format.
Visualization: Annotates the original image with the processed data and generates a summary table.


Setup Instructions
Prerequisites
Ensure you have Python 3.7 or later installed. You will also need to install the following dependencies:

PyTorch
torchvision
transformers
PIL (Pillow)
easyocr
streamlit
pandas
other libraries as listed in requirements.txt


Usage Guidelines
Running the Pipeline
Start the Streamlit Application
(open terminal and write) :- 
1. cd c:/path to project directory/
2. streamlit run app.py


Go to the Streamlit web interface.
Upload an image file (png, jpg, or jpeg).
Process the Image

Start Segmentation: Segment the image and save the results.
Start Object Extraction: Extract objects from the segmented masks.
Start Object Identification: Identify and describe each object.
Start Text Extraction: Extract text from each object.
Start Summarization: Summarize the extracted text.
Start Data Mapping: Map the descriptions, text, and summaries.
Start Visualization: Annotate the original image and generate a summary table.


Output Files

Segmentation Results: Saved in Data/segmentation_output/
Extracted Objects: Saved in Data/segmented_objects/
Descriptions: Saved in Data/object_description.json
Extracted Text: Saved in Data/extracted_text.json
Summaries: Saved in Data/summaries.json
Mapped Data: Saved in Data/mapped_data.json
Annotated Image: Saved in Data/output/
Summary Table: Saved in Data/output/


directory structuture 

IMAGE_SEGMENTATION/
├── Data/
│   ├── input_images/
│   ├── segmentation_output/
│   ├── segmented_objects/
│   ├── FINAL_OUTPUT/
├── Streamlit_app/
│   ├── Model/
│   │   ├── segmentation_model.py
│   │   ├── identification_model.py
│   │   ├── text_extraction_model.py
│   │   ├── summarization_model.py
│   ├── Utils/
│   │   ├── object_extraction.py
│   │   ├── data_mapping.py
│   │   ├── visualization.py
│   ├── app.py
├── requirements.txt
├── README.md