"""Relational store for episodes (R24). SQLAlchemy Core so the same code runs on SQLite (tests,
laptop) and PostgreSQL 18 (Colab, production). Embeddings are JSON here; pgvector columns come
with the retrieval work. Every table is keyed by the deterministic ids the writers already
produce, so loading is idempotent (E-STO-02)."""
from __future__ import annotations

from sqlalchemy import JSON, Boolean, Column, Float, Integer, MetaData, String, Table, Text, create_engine
from sqlalchemy.engine import Engine

metadata = MetaData()

episodes = Table(
    "episodes", metadata,
    Column("episode_id", String, primary_key=True), Column("tile_id", String, index=True),
    Column("camera_ids", JSON), Column("t0_ms", Integer, index=True), Column("t1_ms", Integer),
    Column("status", String), Column("kb_version", Integer), Column("schema_version", String),
)
ticks = Table(
    "ticks", metadata,
    Column("episode_id", String, primary_key=True), Column("camera_id", String, primary_key=True),
    Column("tick_index", Integer, primary_key=True), Column("t_start_ms", Integer, index=True),
    Column("t_end_ms", Integer), Column("modality", String), Column("tubes", JSON),
    Column("event_ids", JSON), Column("scene_state", JSON), Column("gate_energy", Float),
)
tubes = Table(
    "tubes", metadata,
    Column("tube_id", String, primary_key=True), Column("episode_id", String, index=True),
    Column("camera_id", String, index=True), Column("tile_id", String), Column("entity_id", String, index=True),
    Column("named", String), Column("class_label", String, index=True), Column("state", String),
    Column("born_ms", Integer, index=True), Column("last_seen_ms", Integer), Column("box", JSON),
    Column("zone_ids", JSON), Column("modality", String), Column("keyframe_refs", JSON),
    Column("attributes", JSON), Column("merge_candidates", JSON),
)
entities = Table(
    "entities", metadata,
    Column("entity_id", String, primary_key=True), Column("camera_id", String), Column("named", String),
    Column("class_label", String), Column("tube_ids", JSON), Column("first_seen_ms", Integer, index=True),
    Column("last_seen_ms", Integer), Column("best_keyframe_ref", String), Column("embedding", JSON),
)
events = Table(
    "events", metadata,
    Column("event_id", String, primary_key=True), Column("episode_id", String, index=True),
    Column("type", String, index=True), Column("t_ms", Integer, index=True), Column("camera_id", String),
    Column("tile_id", String), Column("zone_id", String, index=True), Column("subject_tube_ids", JSON),
    Column("subject_entity_ids", JSON), Column("object_ids", JSON), Column("payload", JSON),
    Column("confidence", Float), Column("source", String), Column("dedupe_key", String),
)
patches = Table(
    "patches", metadata,
    Column("patch_id", String, primary_key=True), Column("episode_id", String, index=True),
    Column("tube_id", String, index=True), Column("produced_at_ms", Integer), Column("source", String),
    Column("payload", JSON), Column("confidence", Float), Column("modality", String),
    Column("enhanced", Boolean), Column("failed", Boolean),
)
custody = Table(
    "custody", metadata,
    Column("event_id", String, primary_key=True), Column("object_id", String, index=True),
    Column("person_tube_ids", JSON), Column("t_ms", Integer), Column("kind", String), Column("zone_id", String),
)
facts = Table(
    "facts", metadata,
    Column("fact_id", String, primary_key=True), Column("subject", String, index=True), Column("predicate", String),
    Column("object", JSON), Column("status", String), Column("confidence", Float), Column("source", String),
    Column("support", Integer), Column("contradictions", Integer), Column("evidence", JSON),
    Column("first_seen_ms", Integer), Column("last_confirmed_ms", Integer), Column("version", Integer),
)
scripts = Table(
    "scripts", metadata,
    Column("episode_id", String, primary_key=True), Column("text", Text), Column("rendered_at_ms", Integer),
)


def connect(url: str = "sqlite+pysqlite:///:memory:") -> Engine:
    """sqlite+pysqlite:///data/vi.db for a file, postgresql+psycopg://vi:vi@localhost/vi for Postgres."""
    engine = create_engine(url, future=True)
    metadata.create_all(engine)
    return engine


def insert_ignore(conn, table: Table, rows: list[dict]) -> int:
    """Idempotent insert for both dialects (E-STO-02)."""
    if not rows:
        return 0
    if conn.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        stmt = pg_insert(table).values(rows).on_conflict_do_nothing()
    else:
        from sqlalchemy.dialects.sqlite import insert as sq_insert
        stmt = sq_insert(table).values(rows).on_conflict_do_nothing()
    return conn.execute(stmt).rowcount
