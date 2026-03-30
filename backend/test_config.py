#!/usr/bin/env python3
"""
Simple test to verify .env file is being read from root directory.
"""
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

try:
    from config import settings
    
    print("🔍 Configuration Test")
    print("=" * 30)
    print(f"✓ Settings loaded successfully")
    print(f"  Database URL: {settings.database_url}")
    print(f"  Redis URL: {settings.redis_url}")
    print(f"  Model Name: {settings.model_name}")
    print(f"  Max Upload MB: {settings.max_upload_mb}")
    
    # Check if we have actual values from .env
    if "127.0.0.1" in settings.redis_url and settings.redis_url == "redis://127.0.0.1:6379/0":
        print("⚠️  Using default Redis URL - .env file may not be loaded correctly")
        print(f"  Expected: Your Upstash Redis URL")
        print(f"  Got: {settings.redis_url}")
    else:
        print("✓ Custom Redis URL detected from .env")
        
    if "postgres:Datalens90210@127.0.0.1:5432" in settings.database_url:
        print("⚠️  Using default Database URL - .env file may not be loaded correctly")
    else:
        print("✓ Custom Database URL detected from .env")
        
    print("=" * 30)
    print("✓ Configuration test complete")
    
except Exception as e:
    print(f"❌ Error loading configuration: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure .env file exists in the root directory")
    print("2. Check that .env file contains the required variables")
    print(f"3. Expected .env location: {backend_dir.parent}/.env")
