from datetime import timedelta

from sqlalchemy import delete, select

from ..db import get_db
from ..db.models import Chart, Message, ToolRun
from ..db.models import Session as SessionModel
from ..logger import get_logger
from ..utils.time import utcnow

logger = get_logger(__name__)


async def cleanup_expired_sessions() -> int:
    """Delete sessions older than the configured TTL.

    Returns the number of sessions deleted.
    """
    async with get_db() as db:
        result = await db.execute(
            select(SessionModel).where(SessionModel.expires_at < utcnow())
        )
        expired = result.scalars().all()
        if not expired:
            logger.info("No expired sessions to clean up")
            return 0

        logger.info(f"Found {len(expired)} expired sessions to clean up")

        session_ids = [s.session_id for s in expired]

        await db.execute(delete(Message).where(Message.session_id.in_(session_ids)))
        await db.execute(delete(ToolRun).where(ToolRun.session_id.in_(session_ids)))
        await db.execute(delete(Chart).where(Chart.session_id.in_(session_ids)))

        await db.execute(
            delete(SessionModel).where(SessionModel.expires_at < utcnow())
        )

        await db.commit()
        deleted_count = len(expired)

        logger.info(f"Cleaned up {deleted_count} expired sessions")
        return deleted_count


async def cleanup_old_messages(days: int = 30) -> int:
    """Delete messages older than specified days (for all non-expired sessions).

    Returns the number of messages deleted.
    """
    cutoff = utcnow() - timedelta(days=days)

    async with get_db() as db:
        result = await db.execute(select(Message).where(Message.created_at < cutoff))
        old_messages = result.scalars().all()

        if not old_messages:
            logger.info("No old messages to clean up")
            return 0

        logger.info(f"Found {len(old_messages)} messages older than {days} days")

        await db.execute(delete(Message).where(Message.created_at < cutoff))

        await db.commit()
        deleted_count = len(old_messages)

        logger.info(f"Cleaned up {deleted_count} old messages")
        return deleted_count
