import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, List

try:
    from .utils import Vote, get_logger
except ImportError:  # pragma: no cover - script-style fallback for the vendored app
    from utils import Vote, get_logger

if TYPE_CHECKING:
    from pymongo.database import Database
else:
    Database = Any

try:
    from pymongo import MongoClient as _MongoClient
except ImportError:  # pragma: no cover - optional dependency
    _MongoClient = None

logger = get_logger()


def create_db_connection() -> Database:
    if _MongoClient is None:
        raise RuntimeError(
            "JudgeArena database support requires the optional `pymongo` package."
        )
    db = _MongoClient(os.getenv("MONGO_URI")).get_database(os.getenv("MONGO_DB"))
    return db


def add_vote(vote: Vote, db: Database) -> None:
    try:
        db.get_collection("votes").insert_one(vote.__dict__)
        logger.info("Vote added to database")
    except Exception as e:
        logger.error("Error adding vote to database")
        logger.error(e)


def get_votes(db: Database) -> List[Vote]:
    now = datetime.now(timezone.utc)
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    votes = list(
        db.get_collection("votes").find(
            {"timestamp": {"$lte": current_hour.isoformat()}}
        )
    )
    return votes
