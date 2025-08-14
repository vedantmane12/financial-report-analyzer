import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Import your S3FileManager class
from services.s3 import S3FileManager

def upload_directory(local_dir, s3_folder='uploads'):
    """Upload all files from local directory to S3"""
    
    # Initialize S3 manager
    s3_manager = S3FileManager(
        bucket_name=AWS_BUCKET_NAME,
        base_path=s3_folder
    )
    
    # Get all files in directory
    local_path = Path(local_dir)
    files = [f for f in local_path.iterdir() if f.is_file()]
    
    print(f"Uploading {len(files)} files to S3...")
    
    for file_path in files:
        try:
            # Read file content
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Create S3 key
            s3_key = f"{s3_folder}/{file_path.name}"
            
            # Upload file
            s3_manager.upload_file(
                bucket_name=AWS_BUCKET_NAME,
                file_name=s3_key,
                content=content
            )
            
            print(f"✓ Uploaded: {file_path.name}")
            
        except Exception as e:
            print(f"✗ Failed: {file_path.name} - {str(e)}")

# Usage
if __name__ == "__main__":
    # Upload all files from current directory
    upload_directory("financial_filings_data", s3_folder="finance_data_upload")
    