
from sqlalchemy import select

from ..db import get_db as get_db_session
from ..db.models import Message
from ..logger import get_logger

logger = get_logger(__name__)

LLM_CONTEXT_MESSAGES = 10


def _messages_to_langchain(messages: list[Message]) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


async def _summarize_messages(messages: list[Message]) -> str:
    if not messages:
        return "No previous conversation."

    summary_parts = []
    user_msgs = [m for m in messages if m.role == "user"]
    assistant_msgs = [m for m in messages if m.role == "assistant"]

    if user_msgs:
        summary_parts.append(f"User asked {len(user_msgs)} questions")

    if assistant_msgs:
        topics = []
        for msg in assistant_msgs[:3]:
            content = msg.content[:100] if msg.content else ""
            if content:
                topics.append(content)
        if topics:
            summary_parts.append(f"Topics covered: {'; '.join(topics)}")

    return ". ".join(summary_parts) if summary_parts else "Previous conversation available."


async def build_agent_context(session_id: str) -> list[dict]:
    """Build message list for LLM - last N messages + summary if needed"""

    async with get_db_session() as db:
        result = await db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at)
        )
        messages = result.scalars().all()

    if not messages:
        return []

    if len(messages) <= LLM_CONTEXT_MESSAGES:
        return _messages_to_langchain(messages)

    recent = messages[-LLM_CONTEXT_MESSAGES:]
    older = messages[:-LLM_CONTEXT_MESSAGES]

    summary = await _summarize_messages(older)

    context = [
        {"role": "system", "content": f"Previous conversation summary: {summary}"}
    ]
    context.extend(_messages_to_langchain(recent))

    logger.debug(f"Built context for {session_id}: {len(older)} messages summarized, {len(recent)} recent")
    return context


async def get_conversation_summary(session_id: str) -> str:
    """Get a simple text summary of the conversation for short context"""

    async with get_db_session() as db:
        result = await db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.desc())
            .limit(LLM_CONTEXT_MESSAGES)
        )
        messages = result.scalars().all()

    if not messages:
        return "No previous conversation."

    summary_parts = []
    for msg in reversed(messages):
        if msg.role == "user":
            summary_parts.append(f"User: {msg.content[:200]}")
        elif msg.role == "assistant" and msg.content:
            summary_parts.append(f"Assistant: {msg.content[:200]}")

    return "\n".join(summary_parts)
