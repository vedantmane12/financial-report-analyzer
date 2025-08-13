from http.client import HTTPException
import boto3
import json
from datetime import datetime, timedelta
import boto3.s3.transfer as transfer
import os
from dotenv import load_dotenv
load_dotenv()

# Read values
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

class S3FileManager:
    def __init__(self, bucket_name, base_path=''):
        self.s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, 
                               aws_secret_access_key=AWS_SECRET_ACCESS_KEY,)
        self.bucket_name = bucket_name
        self.base_path = base_path.strip('/')
    
    def list_files(self, prefix=''):
        full_prefix = f'{self.base_path}/{prefix}'.strip('/')
        response = self.s3.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=full_prefix
        )
        return [obj['Key'] for obj in response.get('Contents', [])]

    def load_s3_file_content(self, file_name):
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=file_name)
            content = response['Body'].read().decode('utf-8')
            return content
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading file {file_key}: {str(e)}")
    
    def load_s3_pdf(self, file_name):
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=file_name)
            content = response['Body'].read()
            return content
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading file {file_key}: {str(e)}")

    def upload_file(self, bucket_name, file_name, content):
        self.s3.put_object(Bucket=bucket_name, Key=file_name, Body=content)
    
    def get_presigned_url(self, object_name, expiration=3600):
        full_path = f'{self.base_path}/{object_name}'.strip('/')
        try:
            url = self.s3.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': full_path
                },
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            print(f"Error generating presigned URL: {e}")
            return None
    
    def upload_with_retry(self, file_path, bucket_name, object_name=None, max_attempts=3):
        config = transfer.TransferConfig(
            multipart_threshold=1024 * 25,  # 25MB
            max_concurrency=10,
            multipart_chunksize=1024 * 25,
            use_threads=True
        )
        
        for attempt in range(max_attempts):
            try:
                self.s3.upload_file(
                    file_path,
                    bucket_name,
                    object_name or file_path,
                    Config=config
                )
                print(f'File uploaded successfully on attempt {attempt + 1}')
                return True
            except Exception as e:
                if attempt == max_attempts - 1:
                    print(f'Final attempt failed: {str(e)}')
                    return False
                print(f'Attempt {attempt + 1} failed, retrying...')