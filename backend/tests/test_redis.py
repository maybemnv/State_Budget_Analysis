"""Test Redis/Upstash connection using settings from .env"""
import asyncio
import sys
from pathlib import Path

backend_path = str(Path(__file__).parent.parent)
sys.path.insert(0, backend_path)

from config import settings  # noqa: E402


async def test_redis_connection():
    """Test Redis connection (either Upstash or local)"""
    print("=" * 60)
    print("REDIS CONNECTION TEST")
    print("=" * 60)

    # Show configuration (safely - no credentials exposed)
    redis_info = settings.get_redis_info()
    print("\nRedis Configuration:")
    print(f"  Type: {redis_info['type']}")
    print(f"  URL: {redis_info['url']}")
    print(f"  Configured: {redis_info['configured']}")
    print(f"  Using Upstash: {settings.is_upstash_redis}")
    print()

    try:
        # Try Upstash first if configured
        if settings.is_upstash_redis:
            print("Test 1: Testing Upstash Redis connection...")
            try:
                from upstash_redis.asyncio import Redis

                client = Redis(
                    url=settings.upstash_redis_rest_url,
                    token=settings.upstash_redis_rest_token
                )

                # Test ping
                pong = await client.ping()
                if pong:
                    print("  ✓ Upstash Redis connection successful")
                else:
                    print("  ✗ Ping failed")
                    return False

                # Test set/get
                await client.set("test_key", "test_value", ex=10)
                value = await client.get("test_key")
                if value == "test_value":
                    print("  ✓ Set/Get operations working")
                else:
                    print(f"  ✗ Set/Get failed: expected 'test_value', got {value}")
                    return False

                # Clean up
                await client.delete("test_key")
                print("  ✓ Cleanup successful")

                await client.close()

                print("\n" + "=" * 60)
                print("UPSTASH REDIS TEST PASSED ✓")
                print("=" * 60)
                return True

            except ImportError:
                print("  ⚠ upstash-redis package not installed")
                print("  Trying standard redis instead...")
            except Exception as e:
                print(f"  ✗ Upstash connection failed: {str(e)}")
                print("\n  Trying standard redis as fallback...")

        # Try standard Redis
        print("\nTest 2: Testing standard Redis connection...")
        try:
            import redis.asyncio as redis

            client = redis.from_url(settings.redis_url)

            # Test ping
            pong = await client.ping()
            if pong:
                print("  ✓ Redis connection successful")
            else:
                print("  ✗ Ping failed")
                return False

            # Test set/get
            await client.set("test_key", "test_value", ex=10)
            value = await client.get("test_key")
            if value == b"test_value" or value == "test_value":
                print("  ✓ Set/Get operations working")
            else:
                print(f"  ✗ Set/Get failed: expected 'test_value', got {value}")
                return False

            # Clean up
            await client.delete("test_key")
            print("  ✓ Cleanup successful")

            await client.close()

            print("\n" + "=" * 60)
            print("STANDARD REDIS TEST PASSED ✓")
            print("=" * 60)
            return True

        except ImportError:
            print("  ✗ redis package not installed")
            return False
        except Exception as e:
            print(f"  ✗ Redis connection failed: {str(e)}")
            return False

    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        print("\nDetailed traceback:")
        print(traceback.format_exc())
        print("\n" + "=" * 60)
        print("REDIS TEST FAILED ✗")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = asyncio.run(test_redis_connection())
    sys.exit(0 if success else 1)
