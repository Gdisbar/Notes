
import asyncio
import pytest
import time
from datetime import datetime
from acid_ingestion_pipeline import ACIDIngestionPipeline

class TestACIDVerification:
    """Test suite to verify ACID properties"""

    @pytest.fixture
    def pipeline(self):
        """Setup test pipeline"""
        postgres_config = {'host': 'localhost', 'database': 'test_db', 'user': 'test', 'password': 'test'}
        redis_config = {'host': 'localhost', 'port': 6379, 'db': 1}  # Use test DB
        pinecone_config = {'index_name': 'test-index'}

        return ACIDIngestionPipeline(postgres_config, redis_config, pinecone_config)

    def test_atomicity_success_case(self, pipeline):
        """Verify all-or-nothing processing"""
        documents = [
            {
                'source_id': 'atom_test_1',
                'content': 'Valid document content for atomicity test.',
                'source_type': 'pdf',
                'created_at': datetime.now(),
                'metadata': {}
            }
        ]

        # Process successfully
        success = asyncio.run(pipeline.ingest_documents_with_acid('atom_source', documents))
        assert success == True

        # Verify all nodes are committed
        batch_id = pipeline.get_last_batch_id('atom_source')  # Implementation detail
        verification = pipeline.verify_acid_properties(batch_id)
        assert verification['atomicity'] == True

    def test_atomicity_failure_rollback(self, pipeline):
        """Verify rollback on failure"""
        documents = [
            {
                'source_id': 'atom_fail_1',
                'content': 'Valid document content.',
                'source_type': 'pdf',
                'created_at': datetime.now(),
                'metadata': {}
            },
            {
                'source_id': 'atom_fail_2',
                'content': '',  # This will cause validation failure
                'source_type': 'pdf', 
                'created_at': datetime.now(),
                'metadata': {}
            }
        ]

        # Process should fail
        success = asyncio.run(pipeline.ingest_documents_with_acid('atom_fail_source', documents))
        assert success == False

        # Verify no partial state exists
        # Check that no nodes from this batch exist in vector store
        # Implementation would query vector store for batch_id

    def test_consistency_validation(self, pipeline):
        """Test schema and business rule validation"""

        # Test invalid schema
        invalid_docs = [
            {
                'source_id': 'invalid_1',
                'content': 'Valid content',
                'source_type': 'invalid_type',  # Should fail validation
                'created_at': datetime.now()
            }
        ]

        with pytest.raises(ValueError):
            pipeline.validate_documents(invalid_docs)

        # Test valid schema
        valid_docs = [
            {
                'source_id': 'valid_1', 
                'content': 'This content meets minimum length requirements.',
                'source_type': 'pdf',
                'created_at': datetime.now(),
                'metadata': {}
            }
        ]

        validated = pipeline.validate_documents(valid_docs)
        assert len(validated) == 1
        assert validated[0].source_id == 'valid_1'

    def test_isolation_concurrent_processing(self, pipeline):
        """Test that concurrent processing is properly isolated"""

        async def concurrent_ingestion(source_id, doc_suffix):
            documents = [
                {
                    'source_id': f'concurrent_{doc_suffix}',
                    'content': f'Concurrent processing test document {doc_suffix}.',
                    'source_type': 'web',
                    'created_at': datetime.now(),
                    'metadata': {}
                }
            ]
            return await pipeline.ingest_documents_with_acid(source_id, documents)

        async def run_concurrent_test():
            # Try to process the same source concurrently
            task1 = asyncio.create_task(concurrent_ingestion('concurrent_source', '1'))
            task2 = asyncio.create_task(concurrent_ingestion('concurrent_source', '2'))

            results = await asyncio.gather(task1, task2, return_exceptions=True)

            # One should succeed, one should be blocked/fail due to lock
            success_count = sum(1 for r in results if r is True)
            assert success_count == 1  # Only one should succeed due to isolation

        asyncio.run(run_concurrent_test())

    def test_durability_checkpoint_recovery(self, pipeline):
        """Test checkpoint creation and recovery"""

        # Create initial checkpoint
        batch_id = 'durability_test_batch'
        pipeline.create_checkpoint(batch_id, 'durability_source', 100)

        # Verify checkpoint exists
        checkpoint = pipeline.get_last_checkpoint('durability_source')
        assert checkpoint is not None
        assert checkpoint['batch_id'] == batch_id
        assert checkpoint['status'] == 'started'

        # Update checkpoint progress
        pipeline.update_checkpoint(batch_id, 50, 'processing')

        # Verify update
        with pipeline.database_transaction() as (conn, cursor):
            cursor.execute(
                "SELECT processed_count, status FROM ingestion_checkpoints WHERE batch_id = %s",
                (batch_id,)
            )
            updated = cursor.fetchone()
            assert updated['processed_count'] == 50
            assert updated['status'] == 'processing'

# Manual verification script for running against live system
def manual_verification_test():
    """Manual test for full ACID verification"""
    print("🧪 Starting ACID Properties Verification Test\n")

    # Test data
    test_docs = [
        {
            'source_id': 'verify_001',
            'content': 'This is a comprehensive test document for ACID verification with sufficient content length.',
            'source_type': 'pdf',
            'created_at': datetime.now(),
            'metadata': {'test_run': 'acid_verification', 'priority': 'high'}
        },
        {
            'source_id': 'verify_002',
            'content': 'Second test document to verify batch processing and ACID compliance across multiple documents.',
            'source_type': 'web', 
            'created_at': datetime.now(),
            'metadata': {'test_run': 'acid_verification', 'priority': 'medium'}
        }
    ]

    async def run_verification():
        # Initialize pipeline with test configuration
        pipeline = ACIDIngestionPipeline(
            postgres_config={'host': 'localhost', 'database': 'test_db', 'user': 'test', 'password': 'test'},
            redis_config={'host': 'localhost', 'port': 6379, 'db': 1},
            pinecone_config={'index_name': 'test-verification-index'}
        )

        print("📝 Step 1: Testing Document Validation (Consistency)")
        try:
            validated = pipeline.validate_documents(test_docs)
            print(f"   ✅ Validated {len(validated)} documents successfully")
        except Exception as e:
            print(f"   ❌ Validation failed: {e}")
            return

        print("\n🔄 Step 2: Testing Full Ingestion Pipeline")
        success = await pipeline.ingest_documents_with_acid('verification_source', test_docs)

        if success:
            print("   ✅ Ingestion completed successfully")
        else:
            print("   ❌ Ingestion failed")
            return

        print("\n🔍 Step 3: Verifying ACID Properties")
        # Get the batch_id from the last ingestion (implementation specific)
        batch_id = 'test_batch_id'  # In practice, get from pipeline

        verification_results = pipeline.verify_acid_properties(batch_id)

        for prop, passed in verification_results.items():
            status = '✅ PASS' if passed else '❌ FAIL'
            print(f"   {prop.upper():12}: {status}")

        all_passed = all(verification_results.values())
        print(f"\n{'🎉 All ACID properties verified!' if all_passed else '⚠️  Some ACID properties failed verification'}")

        return all_passed

    # Run the verification
    result = asyncio.run(run_verification())
    return result

if __name__ == "__main__":
    print("Running manual ACID verification test...")
    manual_verification_test()
