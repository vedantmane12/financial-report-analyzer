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

# Environment Variables
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
MISTRALAI_API_KEY = os.getenv("MISTRALAI_API_KEY") 

def pdf_mistralocr_converter(pdf_stream: io.BytesIO, base_path, s3_obj):
    client = Mistral(api_key=MISTRALAI_API_KEY)
    pdf_stream.seek(0)
    pdf_bytes = pdf_stream.read()

    # Upload PDF file to Mistral's OCR service
    uploaded_file = client.files.upload(
        file={
            "file_name": "temp",
            "content": pdf_bytes,
        },
        purpose="ocr",
    )

    # Get URL for the uploaded file
    signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

    # Process PDF with OCR, including embedded images
    pdf_response = client.ocr.process(
        document=DocumentURLChunk(document_url=signed_url.url),
        model="mistral-ocr-latest",
        include_image_base64=True
    )

    # base_path += f"mistral/{pdf_path.split('/')[-1].split('.')[0]}/"
    final_md_content = get_combined_markdown(pdf_response, s3_obj, base_path)

    md_file_name = f"{base_path}/extracted_data.md"

    # Upload the markdown file to S3   
    s3_obj.upload_file(s3_obj.bucket_name, md_file_name ,final_md_content.encode('utf-8'))
    
    return md_file_name, final_md_content
    

def replace_images_in_markdown(markdown_str: str, images_dict: dict, s3_obj, base_path) -> str:
    for img_name, base64_str in images_dict.items():
        print(f"Replacing image {img_name}")
        # print(base64_str.split(';')[1].split(',')[1])
        base64_str = base64_str.split(';')[1].split(',')[1]
        image_data = base64.b64decode(base64_str)
        
        element_image_filename = f"{base_path}/images/{img_name.split('.')[0]}.png"
        print(element_image_filename)
        # Convert to JPEG format
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Convert to PNG format
        output_buffer = io.BytesIO()
        image.save(output_buffer, format="PNG")
        output_buffer.seek(0)
        
        s3_obj.upload_file(s3_obj.bucket_name, element_image_filename, output_buffer.read())
        element_image_link = f"https://{s3_obj.bucket_name}.s3.amazonaws.com/{f"{element_image_filename}"}"
        markdown_str = markdown_str.replace(
            f"![{img_name}]({img_name})", f"![{img_name}]({element_image_link})"
        )
    return markdown_str

def get_combined_markdown(ocr_response: OCRResponse, s3_obj, base_path) -> str:
    markdowns: list[str] = []
    # Extract images from page
    for page in ocr_response.pages:
        image_data = {}
        for img in page.images:
            image_data[img.id] = img.image_base64
        # Replace image placeholders with actual images
        markdowns.append(replace_images_in_markdown(page.markdown, image_data, s3_obj, base_path))

    return "\n\n".join(markdowns)

def main():
    # pdf_path = "prototypes/input_docs/FY2025Q1.pdf"  # Hardcoded file path
    base_path = "nvdia/"
    
    s3_obj = S3FileManager(AWS_BUCKET_NAME, base_path)
    files = list({file for file in s3_obj.list_files() if file.endswith('.pdf')})
    for file in files:
        print(file)
        pdf_file = s3_obj.load_s3_pdf(file)
        pdf_bytes = io.BytesIO(pdf_file)
        output_path = f"{s3_obj.base_path}/mistral/{file.split('/')[-1].split('.')[0]}"
        file_name, markdown_content = pdf_mistralocr_converter(pdf_bytes, output_path, s3_obj)

if __name__ == "__main__":
    main()