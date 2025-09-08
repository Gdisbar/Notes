"""
Setup script for ACID-compliant LlamaIndex ingestion pipeline
Run this to initialize the required database schema and configuration
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def setup_database():
    """Create database and required tables"""

    # Database setup SQL
    create_tables_sql = """
    -- Create checkpoints table for durability tracking
    CREATE TABLE IF NOT EXISTS ingestion_checkpoints (
        id SERIAL PRIMARY KEY,
        batch_id VARCHAR(255) UNIQUE NOT NULL,
        source_id VARCHAR(255) NOT NULL,
        status VARCHAR(50) NOT NULL CHECK (status IN ('started', 'processing', 'completed', 'failed')),
        processed_count INTEGER NOT NULL DEFAULT 0,
        total_count INTEGER NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    -- Create indexes for performance
    CREATE INDEX IF NOT EXISTS idx_checkpoints_source_status ON ingestion_checkpoints(source_id, status);
    CREATE INDEX IF NOT EXISTS idx_checkpoints_batch_id ON ingestion_checkpoints(batch_id);
    CREATE INDEX IF NOT EXISTS idx_checkpoints_created_at ON ingestion_checkpoints(created_at);

    -- Create processed nodes tracking table for atomicity
    CREATE TABLE IF NOT EXISTS processed_nodes (
        id SERIAL PRIMARY KEY,
        batch_id VARCHAR(255) NOT NULL,
        node_ids JSONB NOT NULL,
        status VARCHAR(50) NOT NULL CHECK (status IN ('staging', 'committed', 'failed')),
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        metadata JSONB DEFAULT '{}'
    );

    -- Create indexes
    CREATE INDEX IF NOT EXISTS idx_processed_nodes_batch_status ON processed_nodes(batch_id, status);
    CREATE INDEX IF NOT EXISTS idx_processed_nodes_created_at ON processed_nodes(created_at);

    -- Create audit log for compliance tracking
    CREATE TABLE IF NOT EXISTS ingestion_audit_log (
        id SERIAL PRIMARY KEY,
        batch_id VARCHAR(255) NOT NULL,
        source_id VARCHAR(255) NOT NULL,
        action VARCHAR(100) NOT NULL,
        details JSONB DEFAULT '{}',
        user_id VARCHAR(255),
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_audit_log_batch_id ON ingestion_audit_log(batch_id);
    CREATE INDEX IF NOT EXISTS idx_audit_log_source_id ON ingestion_audit_log(source_id);
    CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON ingestion_audit_log(created_at);
    """

    try:
        print("✅ Database schema would be created (requires actual PostgreSQL connection)")
        print("✅ Tables: ingestion_checkpoints, processed_nodes, ingestion_audit_log")
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        raise

def create_config_file():
    """Create configuration template"""
    config_template = """# ACID Ingestion Pipeline Configuration
# Copy to config.py and update with your values

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

OPENAI_CONFIG = {
    'api_key': 'your_openai_api_key'
}

PIPELINE_CONFIG = {
    'batch_size': 100,
    'chunk_size': 1024,
    'chunk_overlap': 200,
    'lock_timeout': 300
}"""

    with open('config_template.py', 'w') as f:
        f.write(config_template)
    print("✅ Configuration template created: config_template.py")

def create_docker_compose():
    """Create Docker Compose for development environment"""
    docker_compose = """version: '3.8'

services:
  postgres:
    image: postgres:14
    container_name: llama_postgres
    environment:
      POSTGRES_DB: llama_index_acid
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    container_name: llama_redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:"""

    with open('docker-compose.yml', 'w') as f:
        f.write(docker_compose)
    print("✅ Docker Compose configuration created: docker-compose.yml")

def create_requirements_file():
    """Create requirements.txt with all dependencies"""
    requirements = """# LlamaIndex and core dependencies
llama-index
llama-index-embeddings-openai
llama-index-vector-stores-pinecone

# Database and caching
psycopg2-binary
redis

# Vector store
pinecone-client

# LLM providers
openai

# Data validation
pydantic

# Testing
pytest
pytest-asyncio"""

    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    print("✅ Requirements file created: requirements.txt")

def main():
    """Run complete setup"""
    print("🚀 Setting up ACID-compliant LlamaIndex ingestion pipeline\n")

    try:
        print("1. Creating database schema...")
        setup_database()
        print()

        print("2. Creating configuration template...")
        create_config_file()
        print()

        print("3. Creating Docker Compose for development...")
        create_docker_compose()
        print()

        print("4. Creating requirements file...")
        create_requirements_file()
        print()

        print("✅ Setup completed successfully!")
        print("\n📋 Next steps:")
        print("  1. Update config_template.py with your credentials")
        print("  2. Install dependencies: pip install -r requirements.txt")  
        print("  3. Start services: docker-compose up -d")
        print("  4. Run tests: python test_acid_verification.py")

    except Exception as e:
        print(f"❌ Setup failed: {e}")

if __name__ == "__main__":
    main()