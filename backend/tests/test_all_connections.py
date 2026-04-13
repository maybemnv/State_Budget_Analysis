"""Comprehensive configuration and connection test (manual diagnostic - not pytest)"""
import asyncio
import sys
from pathlib import Path
import pytest

backend_path = str(Path(__file__).parent.parent)
sys.path.insert(0, backend_path)

from config import settings  # noqa: E402


@pytest.mark.skip(reason="Manual diagnostic test - requires live services")
def test_configuration():
    print("=" * 60)
    print("CONFIGURATION VALIDATION")
    print("=" * 60)

    print("\n✓ Safe Configuration Summary:")
    config = settings.get_safe_config_summary()

    print("\n  Database:")
    print("    Host: {}:{}".format(config['database']['host'], config['database']['port']))
    print("    User: {}".format(config['database']['user']))
    print("    Database: {}".format(config['database']['database']))
    print("    Driver: {}".format(config['database']['driver']))

    print("\n  Redis:")
    print("    Type: {}".format(config['redis']['type']))
    print("    URL: {}".format(config['redis']['url']))
    print("    Configured: {}".format(config['redis']['configured']))

    print("\n  Application:")
    print("    Environment: {}".format(config['environment']))
    print("    Debug: {}".format(config['debug']))
    print(f"    Model: {config['model']}")
    print(f"    CORS Origins: {config['cors_origins']}")
    print(f"    Max Upload: {config['max_upload_mb']}MB")
    print(f"    Session TTL: {config['session_ttl_seconds']}s")

    # Security checks
    print("\n" + "-" * 60)
    print("Security Checks:")

    issues = []

    # Check GEMINI_API_KEY is not placeholder
    if settings.gemini_api_key == "your_google_gemini_api_key_here":
        issues.append("✗ GEMINI_API_KEY is still set to placeholder value")
    else:
        print("✓ GEMINI_API_KEY is set (not placeholder)")

    # Check DB credentials are set
    if settings.db_user and settings.db_password:
        print("✓ Database credentials are set")
    else:
        issues.append("✗ Database credentials are missing")

    # Check Redis is configured
    if settings.is_upstash_redis or settings.redis_url:
        print("✓ Redis is configured")
    else:
        issues.append("✗ Redis is not configured")

    # Check if running in production with debug enabled
    if settings.is_production() and settings.debug:
        issues.append("⚠ WARNING: DEBUG is enabled in production environment")
    else:
        print("✓ Debug mode appropriately configured")

    # Check CORS origins don't include wildcards
    if "*" in settings.cors_origins:
        issues.append("⚠ WARNING: CORS allows all origins (*)")
    else:
        print("✓ CORS origins are specific (no wildcards)")

    if issues:
        print("\n" + "\n".join(issues))
        return len([i for i in issues if i.startswith("✗")]) == 0

    return True


@pytest.mark.skip(reason="Manual diagnostic test - requires live services")
async def test_all_connections():
    """Test all database and Redis connections"""
    print("\n" + "=" * 60)
    print("CONNECTION TESTS")
    print("=" * 60)

    all_passed = True

    # Test 1: PostgreSQL
    print("\nTest 1: PostgreSQL Database")
    try:
        import asyncpg

        conn = await asyncpg.connect(
            user=settings.db_user,
            password=settings.db_password,
            host=settings.db_host,
            port=settings.db_port,
            database=settings.db_name
        )

        version = await conn.fetchval("SELECT version()")
        print(f"  ✓ Connected to {version[:50]}...")

        # Check tables
        tables = await conn.fetch(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        )
        print(f"  ✓ Found {len(tables)} tables")

        await conn.close()
        print("  ✓ Connection closed")

    except Exception as e:
        print(f"  ✗ FAILED: {str(e)}")
        all_passed = False

    # Test 2: Redis/Upstash
    print("\nTest 2: Redis")
    try:
        if settings.is_upstash_redis:
            print("  Using Upstash Redis...")
            try:
                from upstash_redis.asyncio import Redis

                client = Redis(
                    url=settings.upstash_redis_rest_url,
                    token=settings.upstash_redis_rest_token
                )

                pong = await client.ping()
                if pong:
                    print("  ✓ Upstash Redis connection successful")

                # Test operations
                await client.set("test_connection", "ok", ex=10)
                value = await client.get("test_connection")
                if value == "ok":
                    print("  ✓ Redis operations working")

                await client.delete("test_connection")
                await client.close()

            except ImportError:
                print("  ⚠ upstash-redis not installed, trying standard redis")
                raise ImportError("upstash-redis not available")
        else:
            print("  Using standard Redis...")
            import redis.asyncio as redis

            client = redis.from_url(settings.redis_url)
            pong = await client.ping()
            if pong:
                print("  ✓ Redis connection successful")

            await client.set("test_connection", "ok", ex=10)
            value = await client.get("test_connection")
            if value == b"ok" or value == "ok":
                print("  ✓ Redis operations working")

            await client.delete("test_connection")
            await client.close()

    except Exception as e:
        print(f"  ✗ FAILED: {str(e)}")
        all_passed = False

    return all_passed


async def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE CONFIGURATION & CONNECTION TEST")
    print("=" * 60 + "\n")

    # Test configuration
    config_ok = test_configuration()

    # Test connections
    connections_ok = await test_all_connections()

    # Final summary
    print("\n" + "=" * 60)
    if config_ok and connections_ok:
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nYour application is properly configured and connected!")
        return True
    else:
        print("SOME TESTS FAILED ✗")
        print("=" * 60)
        print("\nPlease review the errors above and fix your configuration.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
