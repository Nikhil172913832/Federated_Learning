"""Per-client storage backends (folder or SQLite) to simulate hospital stores."""

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ClientStore:
    backend: str
    root: Path
    partition_id: int
    db_path: Optional[Path] = None
    _conn: Optional[sqlite3.Connection] = None

    def ensure(self) -> "ClientStore":
        self.root.mkdir(parents=True, exist_ok=True)
        if self.backend == "sqlite":
            assert self.db_path is not None
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            with self._conn:
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS artifacts (id INTEGER PRIMARY KEY, name TEXT, path TEXT, meta TEXT)"
                )
        return self

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def save_artifact(self, name: str, path: Path, meta: str = "") -> None:
        if self.backend == "sqlite" and self._conn is not None:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO artifacts(name, path, meta) VALUES(?,?,?)",
                    (name, str(path), meta),
                )

    def checkpoint_path(self, round_idx: int) -> Path:
        ckpt_dir = self.root / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir / f"round_{round_idx}.pt"


def get_client_store(partition_id: int, cfg: dict) -> ClientStore:
    storage_cfg = (cfg or {}).get("storage", {})
    backend = storage_cfg.get("backend", "folder")
    root_dir = Path(storage_cfg.get("root_dir", "./client_stores"))
    sqlite_dir = Path(storage_cfg.get("sqlite_dir", "./client_sqlite"))

    client_root = root_dir / f"client_{partition_id}"
    db_path = None
    if backend == "sqlite":
        db_path = sqlite_dir / f"client_{partition_id}.db"

    store = ClientStore(backend=backend, root=client_root, partition_id=partition_id, db_path=db_path)
    return store.ensure()


