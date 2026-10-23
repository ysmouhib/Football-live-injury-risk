"""Log retention job (docs/07_RULES_ETHICS.md; config ``storage.retention_days``).

The log store writes one partition per match at ``{log_root}/{match_date}/{match_id}``
(docs/03_DATA_CONTRACTS.md §9). Retention deletes whole date partitions whose match date is
more than the configured window before today. Nothing outside the log root and no folder that
is not named by an ISO date is ever touched; running the job is idempotent.
"""

from __future__ import annotations

import logging
import re
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_DATE_DIR = re.compile(r"\d{4}-\d{2}-\d{2}")


def enforce_retention(
    log_root: str | Path, retention_days: int, *, today: date | None = None
) -> list[Path]:
    """Delete log partitions older than ``retention_days``; return the removed date folders.

    A partition ``{log_root}/{YYYY-MM-DD}`` is removed with every match in it when its date is
    more than ``retention_days`` before ``today`` (default: today's UTC date; a partition
    exactly ``retention_days`` old is kept). Folders not named by an ISO date are left alone.
    """
    if retention_days < 1:
        raise ValueError("retention_days must be >= 1")
    root = Path(log_root)
    reference = today or datetime.now(UTC).date()
    removed: list[Path] = []
    if not root.is_dir():
        return removed
    for child in sorted(root.iterdir()):
        if not child.is_dir() or _DATE_DIR.fullmatch(child.name) is None:
            continue
        age = (reference - date.fromisoformat(child.name)).days
        if age > retention_days:
            shutil.rmtree(child)
            removed.append(child)
            logger.info("retention: removed log partition %s (%d days old)", child, age)
    return removed
