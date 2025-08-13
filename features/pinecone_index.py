import openai, os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from features.chunk_strategy import semantic_chunking
load_dotenv()

# from services.s3 import S3FileManager

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def connect_to_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if not pc.has_index(PINECONE_INDEX):
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
            tags={
                "environment": "development"
            }
        )
    index = pc.Index(PINECONE_INDEX)
    return index

def read_markdown_file(file, s3_obj):
    content = s3_obj.load_s3_file_content(file)
    return content

def get_embedding(text):
    """Generates an embedding for the given text using OpenAI."""
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def create_pinecone_vector_store(file, chunks):
    index = connect_to_pinecone_index()
    vectors = []
    file = file.split('/')
    # parser = file[1]
    # identifier = file[2]
    year = file[1]
    quarter = file[2]
    records = 0
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        vectors.append((
            f"id_{year}_{quarter}_chunk_{i}",  # Unique ID
            embedding,  # Embedding vector
            {"year": year, "quarter": quarter, "text": chunk}  # Metadata
        ))
        if len(vectors) >= 20:
            records += len(vectors)
            upsert_vectors(index, vectors)
            # index.upsert(vectors=vectors, namespace=f"{parser}_{chunk_strategy}")
            print(f"Inserted {len(vectors)} chunks into Pinecone.")
            vectors.clear()
    # Store in Pinecone under the correct namespace
    if len(vectors)>0:
        upsert_vectors(index, vectors)
        # index.upsert(vectors=vectors, namespace=f"{parser}_{chunk_strategy}")
        print(f"Inserted {len(vectors)} chunks into Pinecone.")
        records += len(vectors)
    print(f"Inserted {records} chunks into Pinecone.")

def upsert_vectors(index, vectors):
    index.upsert(vectors=vectors, namespace=f"nvdia_quarterly_reports")

def query_pinecone(query: str, top_k: int = 10, year: str = None, quarter: list = None):
    # Search the dense index and rerank the results
    index = connect_to_pinecone_index()
    dense_vector = get_embedding(query)
    
    filter_conditions = {}
    if year is not None:
        filter_conditions["year"] = {"$eq": year}
    if quarter is not None:
        if len(quarter) > 0:
            print(quarter)
            filter_conditions["quarter"] = {"$in": quarter}
    print(filter_conditions)
    results = index.query(
        namespace=f"nvdia_quarterly_reports",
        vector=dense_vector,  # Dense vector embedding
        filter=filter_conditions if filter_conditions else None,  # Sparse keyword match
        top_k=top_k,
        include_metadata=True,  # Include chunk text
    )
    responses = []
    for match in results["matches"]:
        print(f"ID: {match['id']}, Score: {match['score']}")
        # print(f"Chunk: {match['metadata']['text']}\n")
        responses.append(match['metadata']['text'])
        print("=================================================================================")
    return responses

def insert_data_into_pinecone():
    base_path = "nvidia"
    s3_obj = S3FileManager(AWS_BUCKET_NAME, base_path)
    
    year = ['2021', '2022', '2023', '2024', '2025']
    quarter = ['Q1', 'Q2', 'Q3', 'Q4']
    
    for y in year:
        for q in quarter:
            file = f"{base_path}/{y}/{q}/mistral/extracted_data.md"
            print(f"Reading file for {y}-{q}")
            content = read_markdown_file(file, s3_obj)
            if len(content) != 0:
                print(f"Successfully read file for {y}-{q}")
                print("Implementing semantic chunking")
                chunks = semantic_chunking(content, max_sentences=10)
                if len(chunks) != 0:
                    print("Successfully chunked the content")
                    print("Creating Pinecone vector store")
                    create_pinecone_vector_store(file, chunks)
                    print(f"Successfully inserted into Pinecone Vector Store for {y}-{q}")
                else:
                    print(f"Failed to chunk content for {y}-{q}")
            else:
                print(f"Failed to extract content for {y}-{q}")
                    
def main():
    query = "What is the revenue of Nvidia?"
    year = "2025"
    quarter = ['Q4', 'Q1']
    top_k = 10
    # responses = query_pinecone(query, top_k, year = year, quarter = None)
    # print(f"Top {top_k} responses for the query '{query}' are:")
    # for i, response in enumerate(responses):
    #     print(f"{i+1}. {response}")
    #     print("=================================================================================")

if __name__ == "__main__":
    main()
    