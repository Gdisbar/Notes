# LlamaIndex Cheatsheet - Open Source Models & Pinecone

## Quick Setup

https://docs.pinecone.io/integrations/llamaindex

### Installation
```bash
# Core LlamaIndex
pip install llama-index-core

# For Pinecone vector store
pip install llama-index-vector-stores-pinecone
pip install pinecone-client

# For open source models
pip install llama-index-llms-huggingface
pip install llama-index-llms-ollama
pip install llama-index-llms-mistralai
pip install llama-index-embeddings-huggingface

# Optional: For quantization
pip install transformers torch bitsandbytes
```

## Open Source LLM Models

### 1. Mistral via API
```python
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import Settings

# Set up Mistral LLM
llm = MistralAI(
    api_key="your_mistral_api_key",
    model="open-mixtral-8x22b",  # or "mistral-7b-instruct"
    temperature=0.1
)

# Set up Mistral embeddings
embed_model = MistralAIEmbedding(
    api_key="your_mistral_api_key",
    model_name="mistral-embed"
)

Settings.llm = llm
Settings.embed_model = embed_model
```

### 2. Local Models via Ollama
```python
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# First run: ollama pull mistral (or llama3.1, mixtral)
llm = Ollama(
    model="mistral",  # or "llama3.1", "mixtral"
    request_timeout=30.0
)

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

Settings.llm = llm
Settings.embed_model = embed_model
```

### 3. HuggingFace Transformers (Local)
```python
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from transformers import BitsAndBytesConfig
import torch

# 4-bit quantization for memory efficiency
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

llm = HuggingFaceLLM(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
    context_window=3900,
    max_new_tokens=256,
    model_kwargs={"quantization_config": quantization_config},
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    device_map="auto",
)

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

Settings.llm = llm
Settings.embed_model = embed_model
```

### 4. Popular Open Source Embedding Models
```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# BGE models (recommended)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# or BAAI/bge-base-en-v1.5, BAAI/bge-large-en-v1.5

# E5 models
embed_model = HuggingFaceEmbedding(model_name="intfloat/e5-small-v2")

# UAE models
embed_model = HuggingFaceEmbedding(model_name="WhereIsAI/UAE-Large-V1")

# Nomic Embed (fully open source)
from llama_index.embeddings.nomic import NomicEmbedding
embed_model = NomicEmbedding(
    api_key="your_nomic_api_key",
    model_name="nomic-embed-text-v1",
    task_type="search_document"
)
```

## Pinecone Integration

### Basic Setup
```python
from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore

# Initialize Pinecone
pc = PineconeGRPC(api_key="your_pinecone_api_key")

index_name = "your-index-name"

# Create index (if it doesn't exist)
pc.create_index(
    name=index_name,
    dimension=384,  # Match your embedding model dimension
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

# Connect to index
pinecone_index = pc.Index(index_name)

# Create vector store
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
```

### Complete RAG Pipeline
```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext

# Load documents
documents = SimpleDirectoryReader("data/").load_data()

# Create storage context with Pinecone
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create index
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)

# Create query engine
query_engine = index.as_query_engine(similarity_top_k=3)

# Query
response = query_engine.query("Your question here")
print(response)
```

## Common Use Cases

### 1. Document Q&A System
```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import StorageContext

# Setup (using your preferred models from above)
documents = SimpleDirectoryReader("./docs").load_data()
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context
)

query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="tree_summarize"
)

response = query_engine.query("What are the main features of this product?")
```

### 2. Chat Interface
```python
# Create chat engine
chat_engine = index.as_chat_engine(
    chat_mode="condense_question",  # or "context", "react"
    verbose=True
)

# Chat with memory
response = chat_engine.chat("Hello, what can you help me with?")
response = chat_engine.chat("Tell me about the company's revenue")
response = chat_engine.chat("How does that compare to last year?")
```

### 3. Multi-Document Analysis
```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine

# Create individual indices for different document types
financial_index = VectorStoreIndex.from_documents(financial_docs, storage_context=storage_context)
legal_index = VectorStoreIndex.from_documents(legal_docs, storage_context=storage_context)

# Create query engines
financial_engine = financial_index.as_query_engine()
legal_engine = legal_index.as_query_engine()

# Create tools
financial_tool = QueryEngineTool.from_defaults(
    query_engine=financial_engine,
    description="Financial documents and reports"
)
legal_tool = QueryEngineTool.from_defaults(
    query_engine=legal_engine,
    description="Legal documents and contracts"
)

# Sub-question query engine
query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[financial_tool, legal_tool]
)
```

### 4. Data Ingestion Pipeline
```python
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.text_splitter import SemanticSplitterNodeParser

# Create ingestion pipeline
pipeline = IngestionPipeline(
    transformations=[
        SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model,
        ),
        embed_model,  # Your chosen embedding model
    ],
    vector_store=vector_store
)

# Run pipeline
pipeline.run(documents=documents)
```

### 5. Advanced Retrieval
```python
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# Custom retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)

# Post-processor to filter results
processor = SimilarityPostprocessor(similarity_cutoff=0.7)

# Query engine with custom retriever
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[processor],
)
```

## Embeddings Dimensions

| Model | Dimension |
|-------|-----------|
| BAAI/bge-small-en-v1.5 | 384 |
| BAAI/bge-base-en-v1.5 | 768 |
| BAAI/bge-large-en-v1.5 | 1024 |
| intfloat/e5-small-v2 | 384 |
| intfloat/e5-base-v2 | 768 |
| WhereIsAI/UAE-Large-V1 | 1024 |
| mistral-embed | 1024 |
| nomic-embed-text-v1 | 768 |

## Performance Tips

1. **Chunking Strategy**: Use semantic splitters for better context preservation
2. **Embedding Models**: BGE models often provide best quality/performance ratio
3. **Quantization**: Use 4-bit quantization for local models to reduce memory
4. **Batch Processing**: Process documents in batches for large datasets
5. **Caching**: Enable caching for frequently accessed embeddings

## Environment Variables
```bash
export PINECONE_API_KEY="your_pinecone_api_key"
export MISTRAL_API_KEY="your_mistral_api_key"
export NOMIC_API_KEY="your_nomic_api_key"
export HUGGINGFACE_API_TOKEN="your_hf_token"  # Optional for HF models
```

## Common Patterns

### Loading from Various Sources
```python
from llama_index.readers.file import PDFReader
from llama_index.readers.web import SimpleWebPageReader

# PDF files
pdf_reader = PDFReader()
documents = pdf_reader.load_data("document.pdf")

# Web pages
web_reader = SimpleWebPageReader()
documents = web_reader.load_data(["https://example.com"])
```

### Memory Management
```python
from llama_index.core.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
chat_engine = index.as_chat_engine(memory=memory)
```

This cheatsheet covers the essential patterns for using LlamaIndex with open source models and Pinecone. Adjust model choices and configurations based on your specific requirements for performance, accuracy, and resource constraints.