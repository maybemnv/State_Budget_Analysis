import io
from typing import Optional

from minio import Minio
from minio.error import S3Error

from ..config import settings
from ..logging import get_logger

logger = get_logger(__name__)


class MinioClient:
    def __init__(self, client: Minio):
        self.client = client
        self.bucket = settings.minio_bucket

    async def ensure_bucket(self) -> None:
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info(f"Created bucket: {self.bucket}")
        except S3Error as e:
            logger.error(f"Failed to ensure bucket: {e}")
            raise

    async def upload_parquet(
        self, session_id: str, data: io.BytesIO, length: int
    ) -> str:
        object_name = f"datasets/{session_id}.parquet"
        try:
            self.client.put_object(
                self.bucket,
                object_name,
                data,
                length,
                content_type="application/octet-stream",
            )
            logger.info(f"Uploaded parquet: {object_name}")
            return object_name
        except S3Error as e:
            logger.error(f"Failed to upload parquet: {e}")
            raise

    async def download_parquet(self, session_id: str) -> Optional[io.BytesIO]:
        object_name = f"datasets/{session_id}.parquet"
        try:
            response = self.client.get_object(self.bucket, object_name)
            data = io.BytesIO(response.read())
            response.close()
            response.release_conn()
            logger.info(f"Downloaded parquet: {object_name}")
            return data
        except S3Error as e:
            if e.code == "NoSuchKey":
                logger.warning(f"Parquet not found: {object_name}")
                return None
            logger.error(f"Failed to download parquet: {e}")
            raise

    async def delete_parquet(self, session_id: str) -> bool:
        object_name = f"datasets/{session_id}.parquet"
        try:
            self.client.remove_object(self.bucket, object_name)
            logger.info(f"Deleted parquet: {object_name}")
            return True
        except S3Error as e:
            logger.error(f"Failed to delete parquet: {e}")
            return False

    async def exists(self, session_id: str) -> bool:
        object_name = f"datasets/{session_id}.parquet"
        try:
            self.client.stat_object(self.bucket, object_name)
            return True
        except S3Error:
            return False


_minio_client: Optional[MinioClient] = None


def get_minio() -> MinioClient:
    global _minio_client
    if _minio_client is None:
        client = Minio(
            settings.minio_url.replace("http://", "").replace("https://", ""),
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_url.startswith("https://"),
        )
        _minio_client = MinioClient(client)
    return _minio_client
