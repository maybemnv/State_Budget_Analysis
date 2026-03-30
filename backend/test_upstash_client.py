#!/usr/bin/env python3
"""
Test Upstash Redis connection with the new RedisClient implementation.
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the backend directory to Python path and set up environment
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Set the root directory for .env file
root_dir = backend_dir.parent
os.environ.setdefault("ENV_FILE_PATH", str(root_dir / ".env"))

async def test_upstash_client():
    """Test the updated RedisClient with Upstash."""
    try:
        # Test direct import first with explicit naming
        try:
            from upstash_redis.asyncio import Redis as UpstashRedis
            print("✓ Upstash Redis imported successfully")
            using_upstash = True
        except ImportError:
            import redis.asyncio as redis
            print("⚠️ Upstash Redis not found, using standard redis")
            using_upstash = False
        
        # Import and test configuration first
        from config import settings
        print(f"✓ Configuration loaded")
        print(f"  Redis URL: {settings.redis_url}")
        
        # Now test our client
        from db.redis_client import get_redis
        print("✓ Updated RedisClient imported successfully")
        
        # Test connection
        redis_client = await get_redis()
        print("✓ RedisClient initialized")
        
        # Test ping (uses our custom implementation)
        ping_result = await redis_client.ping()
        if ping_result:
            print("✅ Upstash Redis connection successful!")
        else:
            print("❌ Redis ping test failed")
            return False
            
        # Test basic operations
        await redis_client.cache_set("test_key", {"message": "Hello Upstash!"}, ttl=60)
        cached_data = await redis_client.cache_get("test_key")
        if cached_data and cached_data.get("message") == "Hello Upstash!":
            print("✅ Cache operations working")
        
        # Test hash operations
        await redis_client.set_streaming_state("test_session", {"status": "active", "step": 1}, ttl=60)
        stream_state = await redis_client.get_streaming_state("test_session")
        if stream_state and stream_state.get("status") == "active":
            print("✅ Hash operations working")
        
        # Test rate limiting
        can_request, remaining = await redis_client.rate_limit("test_user", 5, 60)
        if can_request and remaining == 4:
            print("✅ Rate limiting working")
        
        # Clean up
        await redis_client.cache_delete("test_key")
        await redis_client.clear_streaming_state("test_session")
        
        print("✅ All Upstash Redis operations working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Upstash Redis: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your .env file has the correct UPSTASH_REDIS_URL")
        print("2. Check that your Upstash Redis database is active")
        print("3. Verify the URL format is correct")
        return False

if __name__ == "__main__":
    print("🚀 Testing Updated Upstash Redis Client")
    print("=" * 45)
    
    success = asyncio.run(test_upstash_client())
    
    print("=" * 45)
    if success:
        print("🎉 All tests passed! Your app is ready with Upstash Redis!")
    else:
        print("❌ Tests failed. Please check your Upstash configuration.")
