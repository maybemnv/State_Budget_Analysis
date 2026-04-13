"""Test PostgreSQL database connection using settings from .env"""
import asyncio
import sys
from pathlib import Path

backend_path = str(Path(__file__).parent.parent)
sys.path.insert(0, backend_path)

from config import settings  # noqa: E402
import asyncpg  # noqa: E402


async def test_db():
    """Test PostgreSQL connection with credentials from .env"""
    print("=" * 60)
    print("POSTGRESQL CONNECTION TEST")
    print("=" * 60)
    
    print("\nDatabase Configuration:")
    print(f"  Host: {settings.db_host}:{settings.db_port}")
    print(f"  User: {settings.db_user[:3]}***")
    print(f"  Database: {settings.db_name}")
    print(f"  Driver: {settings.db_driver}")
    print()
    
    try:
        print("Testing database connection...")
        conn = await asyncpg.connect(
            host=settings.db_host,
            port=settings.db_port,
            user=settings.db_user,
            password=settings.db_password,
            database=settings.db_name,
        )
        
        print("✓ Connection successful")
        
        # Get PostgreSQL version
        version = await conn.fetchval("SELECT version()")
        print(f"✓ PostgreSQL Version: {version[:60]}...")
        
        # Check existing tables
        tables = await conn.fetch(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name"
        )
        
        if tables:
            table_names = [t['table_name'] for t in tables]
            print(f"✓ Found {len(tables)} tables: {', '.join(table_names)}")
        else:
            print("  (No tables found yet - this is normal for fresh database)")
        
        await conn.close()
        print("✓ Connection closed")
        
        print("\n" + "=" * 60)
        print("DATABASE CONNECTION TEST PASSED ✓")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        print("\n" + "=" * 60)
        print("DATABASE CONNECTION TEST FAILED ✗")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = asyncio.run(test_db())
    sys.exit(0 if success else 1)
