"""PostgreSQL-backed LangGraph checkpointer.

Stores graph state snapshots in the `checkpoints`/`checkpoint_writes` tables
via the application's SQLAlchemy async engine — one connection system for the
whole app. State blobs are serialized with LangGraph's own JsonPlusSerializer,
so Pydantic models and message payloads round-trip correctly.

Only async methods are implemented (the API is fully async)."""

from __future__ import annotations

import builtins
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Checkpoint as CheckpointRow
from app.db.models import CheckpointWrite


def _thread(config: RunnableConfig) -> tuple[str | None, str, str | None]:
    configurable = config.get("configurable") or {}
    thread_id = configurable.get("thread_id")
    checkpoint_ns = configurable.get("checkpoint_ns", "")
    checkpoint_id = configurable.get("checkpoint_id")
    return thread_id, checkpoint_ns, checkpoint_id


class PostgresCheckpointSaver(BaseCheckpointSaver):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        super().__init__()
        self._session_factory = session_factory

    # ------------------------------------------------------------------ #
    # Core async API                                                      #
    # ------------------------------------------------------------------ #

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id, checkpoint_ns, _ = _thread(config)
        if thread_id is None:
            return None

        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(CheckpointRow)
                    .where(
                        CheckpointRow.thread_id == thread_id,
                        CheckpointRow.checkpoint_ns == checkpoint_ns,
                    )
                    .order_by(CheckpointRow.created_at.desc(), CheckpointRow.checkpoint_id.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if row is None:
                return None

            checkpoint = self.serde.loads_typed((row.type or "json", row.blob))
            metadata = self._load_metadata(row.metadata_blob)
            pending = await self._load_pending_writes(
                session, thread_id, checkpoint_ns, row.checkpoint_id
            )
            parent_config: RunnableConfig | None = (
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": row.parent_checkpoint_id,
                    }
                }
                if row.parent_checkpoint_id
                else None
            )
            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": row.checkpoint_id,
                    }
                },
                checkpoint=checkpoint,
                metadata=cast(CheckpointMetadata, metadata),
                parent_config=parent_config,
                pending_writes=pending,
            )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        thread_id = ((config or {}).get("configurable") or {}).get("thread_id")
        if thread_id is None:
            return

        async with self._session_factory() as session:
            query = (
                select(CheckpointRow)
                .where(CheckpointRow.thread_id == thread_id)
                .order_by(CheckpointRow.created_at.desc(), CheckpointRow.checkpoint_id.desc())
            )
            if limit is not None:
                query = query.limit(limit)
            rows = (await session.execute(query)).scalars().all()

        for row in rows:
            checkpoint = self.serde.loads_typed((row.type or "json", row.blob))
            metadata = self._load_metadata(row.metadata_blob)
            if filter and not all(metadata.get(k) == v for k, v in filter.items()):
                continue
            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": row.checkpoint_ns,
                        "checkpoint_id": row.checkpoint_id,
                    }
                },
                checkpoint=checkpoint,
                metadata=cast(CheckpointMetadata, metadata),
                parent_config=None,
                pending_writes=None,
            )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict,
    ) -> RunnableConfig:
        thread_id, checkpoint_ns, parent_checkpoint_id = _thread(config)

        ckpt_type, ckpt_blob = self.serde.dumps_typed(checkpoint)
        meta_blob = json.dumps(_clean_metadata(metadata)).encode()

        async with self._session_factory() as session:
            session.add(
                CheckpointRow(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint["id"],
                    parent_checkpoint_id=parent_checkpoint_id,
                    type=ckpt_type,
                    blob=ckpt_blob,
                    metadata_blob=meta_blob,
                )
            )
            await session.commit()
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id, checkpoint_ns, checkpoint_id = _thread(config)
        if checkpoint_id is None:
            return

        async with self._session_factory() as session:
            for idx, (channel, value) in enumerate(writes):
                wtype, wblob = self.serde.dumps_typed(value)
                session.add(
                    CheckpointWrite(
                        thread_id=thread_id,
                        checkpoint_ns=checkpoint_ns,
                        checkpoint_id=checkpoint_id,
                        task_id=task_id,
                        idx=idx,
                        channel=channel,
                        type=wtype,
                        blob=wblob,
                    )
                )
            await session.commit()

    async def adelete_thread(self, thread_id: str) -> None:
        async with self._session_factory() as session:
            await session.execute(delete(CheckpointRow).where(CheckpointRow.thread_id == thread_id))
            await session.execute(
                delete(CheckpointWrite).where(CheckpointWrite.thread_id == thread_id)
            )
            await session.commit()

    # ------------------------------------------------------------------ #
    # Sync counterparts (not supported; the app is async-only)            #
    # ------------------------------------------------------------------ #

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        raise NotImplementedError("PostgresCheckpointSaver is async-only")

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict,
    ) -> RunnableConfig:
        raise NotImplementedError("PostgresCheckpointSaver is async-only")

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        raise NotImplementedError("PostgresCheckpointSaver is async-only")

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Any:
        raise NotImplementedError("PostgresCheckpointSaver is async-only")

    def delete_thread(self, thread_id: str) -> None:
        raise NotImplementedError("PostgresCheckpointSaver is async-only")

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _load_metadata(self, blob: bytes) -> dict:
        try:
            return json.loads(blob.decode())
        except (ValueError, UnicodeDecodeError):
            return {}

    async def _load_pending_writes(
        self, session: AsyncSession, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> builtins.list[tuple[str, str, Any]]:
        rows = (
            (
                await session.execute(
                    select(CheckpointWrite)
                    .where(
                        CheckpointWrite.thread_id == thread_id,
                        CheckpointWrite.checkpoint_ns == checkpoint_ns,
                        CheckpointWrite.checkpoint_id == checkpoint_id,
                    )
                    .order_by(CheckpointWrite.task_id, CheckpointWrite.idx)
                )
            )
            .scalars()
            .all()
        )
        pending: list[tuple[str, str, Any]] = []
        for row in rows:
            value = self.serde.loads_typed((row.type or "json", row.blob)) if row.blob else None
            pending.append((row.task_id, row.channel, value))
        return pending


def _clean_metadata(metadata: CheckpointMetadata) -> dict:
    """Keep only JSON-safe, non-sensitive metadata keys."""
    safe: dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if key in {"write_uuids", "parents"}:
            continue
        try:
            json.dumps(value)
            safe[str(key)] = value
        except (TypeError, ValueError):
            safe[str(key)] = str(value)
    return safe
