from datetime import datetime
def utcnow() -> datetime:
    """Return the current UTC time as a timezone-naive datetime.

    Matches the database columns defined as TIMESTAMP WITHOUT TIME ZONE.
    """
    return datetime.utcnow()