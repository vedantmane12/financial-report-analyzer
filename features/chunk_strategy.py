import spacy
import spacy.cli
import tiktoken

TOKEN_LIMIT = 8192  # OpenAI's embedding model limit
SUB_CHUNK_SIZE = 2000  # Safe sub-chunk size to avoid exceeding limits

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    
# Load tokenizer for token counting
tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")

def split_chunk(chunk, max_tokens=SUB_CHUNK_SIZE):
    """Splits a chunk into smaller sub-chunks if it exceeds max_tokens."""
    tokens = tokenizer.encode(chunk)
    if len(tokens) <= max_tokens:
        return [chunk]  # Already within the limit

    sub_chunks = []
    for i in range(0, len(tokens), max_tokens):
        sub_tokens = tokens[i:i + max_tokens]
        sub_chunks.append(tokenizer.decode(sub_tokens))
    
    return sub_chunks

def semantic_chunking(text, max_sentences=5):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
   
    chunks = []
    current_chunk = []
    for i, sent in enumerate(sentences):
        current_chunk.append(sent)
        if (i + 1) % max_sentences == 0:
            merged_chunk = " ".join(current_chunk)
            chunks.extend(split_chunk(merged_chunk, max_tokens=TOKEN_LIMIT // 2))  # Ensure token limits
            current_chunk = []
   
    if current_chunk:
        merged_chunk = " ".join(current_chunk)
        chunks.extend(split_chunk(merged_chunk, max_tokens=TOKEN_LIMIT // 2))
   
    return chunks