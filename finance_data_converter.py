import os, io, json, base64
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from PIL import Image
from mistralai import Mistral
from mistralai import DocumentURLChunk
from mistralai.models import OCRResponse

# from services import s3
from services.s3 import S3FileManager

# Mistral PDF OCR Extractor and Uploader to S3
from features.mistralocr_pdf_extractor import pdf_mistralocr_converter

# Environment Variables
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
MISTRALAI_API_KEY = os.getenv("MISTRALAI_API_KEY")

def main():
    base_path = "finance_data_upload/"
    s3_obj = S3FileManager(AWS_BUCKET_NAME, base_path)
    files = list({file for file in s3_obj.list_files() if file.endswith('.pdf')})
    print(f"Found {len(files)} PDF files to process.")
    for file_path in files:
        try:
            file = file_path.split('/')[-1]
            org = file.split('_')[0]
            year = file.split('_')[1]
            quarter = file.split('_')[2].split('.')[0]
            markdown_base_path = f"finance_data/{org}/{year}/{quarter}"
            print(f"Processing file: {file_path}")
            print(f"Organization: {org}, Year: {year}, Quarter: {quarter}")
            pdf_bytes = s3_obj.load_s3_pdf(file_path)
            pdf_stream = io.BytesIO(pdf_bytes)
            md_file_name, final_md_content = pdf_mistralocr_converter(pdf_stream, markdown_base_path, s3_obj)
            print(f"Extracted data saved to: {md_file_name}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    main()