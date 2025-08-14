import openai, os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from features.chunk_strategy import semantic_chunking
load_dotenv()

from services.s3 import S3FileManager

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
    org = file[1]
    year = file[2]
    quarter = file[3]
    records = 0
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        vectors.append((
            f"id_{org}_{year}_{quarter}_chunk_{i}",  # Unique ID
            embedding,  # Embedding vector
            {"org": org, "year": year, "quarter": quarter, "text": chunk}  # Metadata
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
    index.upsert(vectors=vectors, namespace=f"finance_data")

def query_pinecone(query: str, top_k: int = 20, org: list = None, year: str = None, quarter: list = None):
    # Search the dense index and rerank the results
    index = connect_to_pinecone_index()
    dense_vector = get_embedding(query)
    responses = []
    filter_conditions = {}
    if year is not None:
        filter_conditions["year"] = {"$eq": year}
    if quarter is not None:
        if len(quarter) > 0:
            filter_conditions["quarter"] = {"$in": quarter}
    if org is not None:
        for o in org:
            filter_conditions["org"] = {"$in": [o]}
            print(filter_conditions)
            results=index.query(
                        namespace=f"finance_data",
                        vector=dense_vector,  # Dense vector embedding
                        filter=filter_conditions if filter_conditions else None,  # Sparse keyword match
                        top_k=top_k,
                        include_metadata=True,  # Include chunk text
                    )
            for match in results["matches"]:
                print(f"ID: {match['id']}, Score: {match['score']}")
                responses.append(match['metadata']['text'])
    else:
        results=index.query(
                    namespace=f"finance_data",
                    vector=dense_vector,  # Dense vector embedding
                    filter=filter_conditions if filter_conditions else None,  # Sparse keyword match
                    top_k=top_k,
                    include_metadata=True,  # Include chunk text
                )
        for match in results["matches"]:
            print(f"ID: {match['id']}, Score: {match['score']}")
            responses.append(match['metadata']['text'])
    return responses

