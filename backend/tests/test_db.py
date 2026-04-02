import asyncio
import asyncpg


async def test_db():
    try:
        conn = await asyncpg.connect(
            host="127.0.0.1",
            port=5432,
            user="postgres",
            password="postgres",
            database="datalens",
        )
        print("SUCCESS: Connected to PostgreSQL")
        await conn.close()
    except Exception as e:
        print(f"FAILED: {e}")


asyncio.run(test_db())
