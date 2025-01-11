import os
import json
import cv2
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from pdf2image import convert_from_path
import numpy as np
# Load environment variables
load_dotenv(r"D:\Hackathon 2\Variable.env")

# Get API key and folder paths
LLAMAPARSE_API_KEY = os.getenv('LLAMAPARSE_API_KEY')
PDF_FOLDER_PATH = os.getenv('PDF_FOLDER_PATH', './pdfs')
OUTPUT_FOLDER_PATH = os.getenv('OUTPUT_FOLDER_PATH', './extracted')

# Set up parser
parser = LlamaParse(
    result_type="markdown",
    api_key=LLAMAPARSE_API_KEY
)

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)

# Function to detect and extract images/diagrams from a page using OpenCV
def extract_images_from_page(image, output_base_path, page_idx, doc_idx):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_count = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 50:  # Filter by size
            image_crop = image[y:y+h, x:x+w]
            output_image_path = os.path.join(OUTPUT_FOLDER_PATH,f"{os.path.splitext(filename)[0]}_doc{doc_idx}_page_{page_idx}_image_{image_count}.png")
            cv2.imwrite(output_image_path, image_crop)
            image_count += 1

def process_pdf_document(pdf_path, filename):
    print(f"Processing: {filename}")
    documents = SimpleDirectoryReader(input_files=[pdf_path], file_extractor={".pdf": parser}).load_data()

    if isinstance(documents, list):
        for idx, doc in enumerate(documents):
            output_base_path = os.path.join(OUTPUT_FOLDER_PATH, f"{os.path.splitext(filename)[0]}_doc{idx}")

            # Save extracted text
            if hasattr(doc, 'text'):
                with open(f"{output_base_path}_text.txt", "w", encoding='utf-8') as text_file:
                    text_file.write(doc.text)

            # For tables
            if hasattr(doc, 'tables') and doc.tables:  # Check if tables exist
                print(f"Found {len(doc.tables)} tables.")
                for i, table in enumerate(doc.tables):
                    if isinstance(table, dict):  # Check if table is a dictionary
                        with open(f"{output_base_path}_table_{i}.json", "w", encoding='utf-8') as table_file:
                            json.dump(table, table_file, indent=2, ensure_ascii=False)
                            print("Table stored")
                    else:
                        print(f"Table {i} is not a valid dictionary.")
            else:
                print("No tables found in document.")

    # Extract images from the pages
    extract_images_from_pdf(pdf_path, output_base_path)

    print(f"Extraction complete for: {filename}")

def extract_images_from_pdf(pdf_path, output_base_path):
    pages = convert_from_path(pdf_path)  # Convert PDF to images

    # # Ensure output directory for images exists
    # os.makedirs(output_base_path, exist_ok=True)

    for page_idx, page_image in enumerate(pages):
        # Save the full page as an image (optional)
        # output_page_path = os.path.join(output_base_path, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{page_idx}.png")
        # page_image.save(output_page_path, 'PNG')

        # Read the saved image with OpenCV
        image = np.array(page_image)

        # Call the function to extract images/diagrams from the page
        extract_images_from_page(image, output_base_path, page_idx, 0)  # doc_idx set to 0 since we're processing one PDF at a time

# Process all PDF files in the specified folder
for filename in os.listdir(PDF_FOLDER_PATH):
    if filename.lower().endswith('.pdf'):
        pdf_path = os.path.join(PDF_FOLDER_PATH, filename)
        try:
            process_pdf_document(pdf_path, filename)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")