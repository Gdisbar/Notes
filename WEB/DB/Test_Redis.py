import os
import asyncio
import redis.asyncio as async_redis
import redis
from dotenv import load_dotenv

load_dotenv()

async def test_new_redis_config():
    """Quick test for your new Redis Cloud database"""
    
    # Your exact Redis Cloud configuration
    redis_host = os.getenv("Redis_endpoint", "").split(":")[0]
    redis_port = int(os.getenv("Redis_port", "17037"))
    redis_username = "default"
    redis_password = os.getenv("Redis_password")
    
    print("🧪 Testing New Redis Cloud Database")
    print("=" * 40)
    print(f"Host: {redis_host}")
    print(f"Port: {redis_port}")
    print(f"Username: {redis_username}")
    print(f"Password: {'✅ Set' if redis_password else '❌ Missing'}")
    print()
    
    try:
        # Test async client (non-SSL as per your dashboard)
        async_client = async_redis.Redis(
            host=redis_host,
            port=redis_port,
            username=redis_username,
            password=redis_password,
            decode_responses=True,
            ssl=False  # Non-SSL based on your connection example
        )
        
        print("🔄 Testing async Redis client...")
        success = await async_client.set('foo', 'bar')
        result = await async_client.get('foo')
        print(f"✅ Async test - Set: {success}, Get: {result}")
        
        # Test sync client
        sync_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            username=redis_username,
            password=redis_password,
            decode_responses=True,
            ssl=False
        )
        
        print("🔄 Testing sync Redis client...")
        sync_success = sync_client.set('sync_test', 'working')
        sync_result = sync_client.get('sync_test')
        print(f"✅ Sync test - Set: {sync_success}, Get: {sync_result}")
        
        # Test hash operations (used in your app)
        print("🔄 Testing hash operations...")
        await async_client.hset('test_hash', mapping={'status': 'queued', 'progress': '0'})
        hash_data = await async_client.hgetall('test_hash')
        print(f"✅ Hash test: {hash_data}")
        
        # Cleanup
        await async_client.delete('foo', 'test_hash')
        sync_client.delete('sync_test')
        
        await async_client.aclose()
        
        print()
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your Redis configuration is working correctly")
        print("✅ Ready to run your FastAPI application")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_new_redis_config())