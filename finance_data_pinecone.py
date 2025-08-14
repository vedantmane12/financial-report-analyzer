import openai, os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()

from services.s3 import S3FileManager
from features.pinecone_index import read_markdown_file, create_pinecone_vector_store
from features.chunk_strategy import semantic_chunking

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    # index = connect_to_pinecone_index()
    base_path = "finance_data/"
    s3_obj = S3FileManager(AWS_BUCKET_NAME, base_path)
    org = ["AMD", "INTEL", "NVIDIA"]
    year = ["2023", "2024", "2025"]
    quarter = ["Q1", "Q2", "Q3", "Q4"]
    # files = list({file for file in s3_obj.list_files() if file.endswith('.md')})
    file_path = [f"{base_path}{o}/{y}/{q}/extracted_data.md" for o in org for y in year for q in quarter]
    print(f"Found {len(file_path)} PDF files to process.")
    for file in file_path:
        try:
            print(f"Processing file: {file}")
            content = read_markdown_file(file, s3_obj)
            if len(content) != 0:
                org = file.split('/')[1]
                year = file.split('/')[2]
                quarter = file.split('/')[3]
                print(f"Organization: {org}, Year: {year}, Quarter: {quarter}")
                print("Implementing chunking...")
                chunks = semantic_chunking(content, max_sentences=10)
                if len(chunks) != 0:
                    print("Successfully chunked the content")
                    print("Chunks created:", len(chunks))
                    print("Creating Pinecone vector store")
                    try:
                        create_pinecone_vector_store(file, chunks)
                    except Exception as e:
                        print(f"Error inserting into Pinecone Vector Store: {str(e)}")
                    print(f"Successfully inserted into Pinecone Vector Store for {year}-{quarter} of {org}")
                else:
                    print(f"Failed to chunk content for {year}-{quarter} of {org}")
                # print(f"Successfully read file for {year}-{quarter} of {org}")
            else:
                print("Failed to read content from the file.")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    main()