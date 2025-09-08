# ACID Ingestion Pipeline Configuration
DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'llama_index_acid',
    'user': 'your_db_user',
    'password': 'your_db_password'
}

REDIS_CONFIG = {
    'host': 'localhost', 
    'port': 6379,
    'db': 0
}

PINECONE_CONFIG = {
    'api_key': 'your_pinecone_api_key',
    'index_name': 'document-index'
}

PIPELINE_CONFIG = {
    'batch_size': 100,
    'chunk_size': 1024,
    'chunk_overlap': 200,
    'lock_timeout': 300
}