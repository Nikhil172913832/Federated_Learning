"""Storage system for client-side personalized parameters.

Handles per-client storage of adapters, LoRA matrices, and personalized layers
using SQLite for metadata and filesystem for parameter checkpoints.
"""

import json
import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any, List
import torch


@dataclass
class PersonalizationMetadata:
    """Metadata for a personalized model component."""
    
    client_id: int
    component_type: str  # "adapter", "lora", "head", "full"
    architecture_name: str
    param_count: int
    created_round: int
    last_updated_round: int
    version: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "component_type": self.component_type,
            "architecture_name": self.architecture_name,
            "param_count": self.param_count,
            "created_round": self.created_round,
            "last_updated_round": self.last_updated_round,
            "version": self.version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonalizationMetadata":
        return cls(**data)


class PersonalizationStore:
    """Manages storage and retrieval of personalized model parameters per client."""
    
    def __init__(
        self,
        client_id: int,
        storage_root: Path,
        backend: str = "sqlite",
    ):
        """Initialize personalization storage.
        
        Args:
            client_id: Unique client identifier
            storage_root: Root directory for storage
            backend: Storage backend ("sqlite" or "file")
        """
        self.client_id = client_id
        self.storage_root = Path(storage_root)
        self.backend = backend
        
        # Create client-specific directories
        self.client_dir = self.storage_root / f"client_{client_id}"
        self.params_dir = self.client_dir / "personalized_params"
        self.checkpoints_dir = self.client_dir / "checkpoints"
        
        self.client_dir.mkdir(parents=True, exist_ok=True)
        self.params_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # SQLite database for metadata
        self.db_path = self.client_dir / f"personalization_{client_id}.db"
        self._conn: Optional[sqlite3.Connection] = None
        
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """Initialize SQLite database schema."""
        self._conn = sqlite3.connect(str(self.db_path))
        
        with self._conn:
            # Metadata table
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS personalization_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id INTEGER NOT NULL,
                    component_type TEXT NOT NULL,
                    architecture_name TEXT NOT NULL,
                    param_count INTEGER,
                    created_round INTEGER,
                    last_updated_round INTEGER,
                    version INTEGER,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Parameter files table
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS parameter_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metadata_id INTEGER,
                    file_path TEXT NOT NULL,
                    round_idx INTEGER,
                    file_size_bytes INTEGER,
                    checksum TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (metadata_id) REFERENCES personalization_metadata(id)
                )
            """)
            
            # Training history table
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS training_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id INTEGER NOT NULL,
                    round_idx INTEGER NOT NULL,
                    global_loss REAL,
                    personalized_loss REAL,
                    kd_loss REAL,
                    personalization_gain REAL,
                    num_samples INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def save_personalized_params(
        self,
        params: Dict[str, torch.Tensor],
        component_type: str,
        architecture_name: str,
        round_idx: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Save personalized parameters to storage.
        
        Args:
            params: Dictionary of parameter tensors
            component_type: Type of component ("adapter", "lora", "head")
            architecture_name: Name of the architecture
            round_idx: Current federated learning round
            metadata: Optional additional metadata
            
        Returns:
            Metadata ID
        """
        # Count parameters
        param_count = sum(p.numel() for p in params.values())
        
        # Generate file path
        filename = f"{component_type}_round_{round_idx}_v{self._get_next_version()}.pt"
        file_path = self.params_dir / filename
        
        # Save parameters to disk
        torch.save(params, file_path)
        file_size = file_path.stat().st_size
        
        # Check if metadata exists
        with self._conn:
            cursor = self._conn.execute(
                """SELECT id, version FROM personalization_metadata 
                   WHERE client_id = ? AND component_type = ? AND architecture_name = ? 
                   AND is_active = 1""",
                (self.client_id, component_type, architecture_name)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Update existing
                metadata_id, old_version = existing
                new_version = old_version + 1
                
                self._conn.execute(
                    """UPDATE personalization_metadata 
                       SET last_updated_round = ?, version = ? 
                       WHERE id = ?""",
                    (round_idx, new_version, metadata_id)
                )
            else:
                # Insert new
                cursor = self._conn.execute(
                    """INSERT INTO personalization_metadata 
                       (client_id, component_type, architecture_name, param_count, 
                        created_round, last_updated_round, version)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (self.client_id, component_type, architecture_name, param_count,
                     round_idx, round_idx, 1)
                )
                metadata_id = cursor.lastrowid
            
            # Insert parameter file record
            self._conn.execute(
                """INSERT INTO parameter_files 
                   (metadata_id, file_path, round_idx, file_size_bytes)
                   VALUES (?, ?, ?, ?)""",
                (metadata_id, str(file_path), round_idx, file_size)
            )
        
        return metadata_id
    
    def load_personalized_params(
        self,
        component_type: str,
        architecture_name: str,
        round_idx: Optional[int] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Load personalized parameters from storage.
        
        Args:
            component_type: Type of component to load
            architecture_name: Name of the architecture
            round_idx: Specific round to load (None = latest)
            
        Returns:
            Parameter dictionary or None if not found
        """
        with self._conn:
            if round_idx is not None:
                cursor = self._conn.execute(
                    """SELECT pf.file_path FROM parameter_files pf
                       JOIN personalization_metadata pm ON pf.metadata_id = pm.id
                       WHERE pm.client_id = ? AND pm.component_type = ? 
                       AND pm.architecture_name = ? AND pf.round_idx = ?
                       AND pm.is_active = 1
                       ORDER BY pf.created_at DESC LIMIT 1""",
                    (self.client_id, component_type, architecture_name, round_idx)
                )
            else:
                cursor = self._conn.execute(
                    """SELECT pf.file_path FROM parameter_files pf
                       JOIN personalization_metadata pm ON pf.metadata_id = pm.id
                       WHERE pm.client_id = ? AND pm.component_type = ? 
                       AND pm.architecture_name = ?
                       AND pm.is_active = 1
                       ORDER BY pf.round_idx DESC, pf.created_at DESC LIMIT 1""",
                    (self.client_id, component_type, architecture_name)
                )
            
            result = cursor.fetchone()
            
        if result is None:
            return None
        
        file_path = Path(result[0])
        if not file_path.exists():
            return None
        
        return torch.load(file_path)
    
    def log_training_metrics(
        self,
        round_idx: int,
        global_loss: float,
        personalized_loss: float,
        kd_loss: float,
        personalization_gain: float,
        num_samples: int,
    ) -> None:
        """Log training metrics for a round.
        
        Args:
            round_idx: Federated learning round
            global_loss: Loss using only global model
            personalized_loss: Loss using personalized model
            kd_loss: Knowledge distillation loss component
            personalization_gain: Accuracy gain from personalization
            num_samples: Number of training samples
        """
        with self._conn:
            self._conn.execute(
                """INSERT INTO training_history 
                   (client_id, round_idx, global_loss, personalized_loss, 
                    kd_loss, personalization_gain, num_samples)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (self.client_id, round_idx, global_loss, personalized_loss,
                 kd_loss, personalization_gain, num_samples)
            )
    
    def get_training_history(
        self,
        start_round: int = 0,
        end_round: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve training history.
        
        Args:
            start_round: Starting round (inclusive)
            end_round: Ending round (inclusive, None = all)
            
        Returns:
            List of training history records
        """
        with self._conn:
            if end_round is not None:
                cursor = self._conn.execute(
                    """SELECT * FROM training_history 
                       WHERE client_id = ? AND round_idx >= ? AND round_idx <= ?
                       ORDER BY round_idx ASC""",
                    (self.client_id, start_round, end_round)
                )
            else:
                cursor = self._conn.execute(
                    """SELECT * FROM training_history 
                       WHERE client_id = ? AND round_idx >= ?
                       ORDER BY round_idx ASC""",
                    (self.client_id, start_round)
                )
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def save_checkpoint(
        self,
        round_idx: int,
        global_params: Dict[str, torch.Tensor],
        personalized_params: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a complete checkpoint (global + personalized).
        
        Args:
            round_idx: Current round
            global_params: Global model parameters
            personalized_params: Personalized parameters
            metadata: Optional metadata
            
        Returns:
            Path to checkpoint file
        """
        checkpoint = {
            "round_idx": round_idx,
            "client_id": self.client_id,
            "global_params": global_params,
            "personalized_params": personalized_params,
            "metadata": metadata or {},
        }
        
        ckpt_path = self.checkpoints_dir / f"checkpoint_round_{round_idx}.pt"
        torch.save(checkpoint, ckpt_path)
        
        return ckpt_path
    
    def load_checkpoint(
        self,
        round_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """Load a complete checkpoint.
        
        Args:
            round_idx: Round to load
            
        Returns:
            Checkpoint dictionary or None
        """
        ckpt_path = self.checkpoints_dir / f"checkpoint_round_{round_idx}.pt"
        
        if not ckpt_path.exists():
            return None
        
        return torch.load(ckpt_path)
    
    def get_latest_round(self) -> Optional[int]:
        """Get the latest round with saved parameters.
        
        Returns:
            Latest round number or None
        """
        with self._conn:
            cursor = self._conn.execute(
                """SELECT MAX(round_idx) FROM training_history 
                   WHERE client_id = ?""",
                (self.client_id,)
            )
            result = cursor.fetchone()
        
        return result[0] if result and result[0] is not None else None
    
    def _get_next_version(self) -> int:
        """Get next version number for parameter files."""
        with self._conn:
            cursor = self._conn.execute(
                """SELECT MAX(version) FROM personalization_metadata 
                   WHERE client_id = ?""",
                (self.client_id,)
            )
            result = cursor.fetchone()
        
        return (result[0] or 0) + 1 if result else 1
    
    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_personalization_store(
    client_id: int,
    config: Dict[str, Any],
) -> PersonalizationStore:
    """Factory function to create a personalization store.
    
    Args:
        client_id: Client identifier
        config: Configuration dictionary
        
    Returns:
        PersonalizationStore instance
    """
    personalization_cfg = config.get("personalization", {})
    storage_cfg = personalization_cfg.get("storage", {})
    
    storage_root = Path(storage_cfg.get("root_dir", "./personalization_store"))
    backend = storage_cfg.get("backend", "sqlite")
    
    return PersonalizationStore(
        client_id=client_id,
        storage_root=storage_root,
        backend=backend,
    )
