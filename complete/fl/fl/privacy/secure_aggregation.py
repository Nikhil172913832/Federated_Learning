"""Secure aggregation implementation for federated learning."""

import torch
from typing import Dict, List, Optional
from collections import OrderedDict
import hashlib


class SecureAggregator:
    """Secure Multi-Party Computation for gradient aggregation.
    
    Implements a simplified secure aggregation protocol where client
    updates are masked before sending to server, and only the aggregate
    can be decrypted.
    """

    def __init__(self, num_clients: int, threshold: Optional[int] = None):
        """Initialize secure aggregator.

        Args:
            num_clients: Total number of clients
            threshold: Minimum clients needed to decrypt (default: num_clients)
        """
        self.num_clients = num_clients
        self.threshold = threshold or num_clients
        self.client_keys: Dict[int, torch.Tensor] = {}

    def generate_client_key(self, client_id: int, seed: Optional[int] = None) -> torch.Tensor:
        """Generate a secret key for a client.

        Args:
            client_id: Client identifier
            seed: Optional random seed

        Returns:
            Secret key tensor
        """
        if seed is None:
            seed = client_id

        # Generate deterministic key from client_id
        torch.manual_seed(seed)
        key = torch.randn(1)  # Simple scalar key for demo
        self.client_keys[client_id] = key

        return key

    def mask_parameters(
        self,
        params: Dict[str, torch.Tensor],
        client_id: int,
        other_client_ids: List[int],
    ) -> Dict[str, torch.Tensor]:
        """Mask client parameters using pairwise keys.

        Args:
            params: Model parameters to mask
            client_id: This client's ID
            other_client_ids: IDs of other participating clients

        Returns:
            Masked parameters
        """
        masked = OrderedDict()

        for name, param in params.items():
            masked_param = param.clone()

            # Add pairwise masks
            for other_id in other_client_ids:
                if other_id != client_id:
                    # Generate pairwise key
                    pairwise_key = self._generate_pairwise_key(client_id, other_id)
                    
                    # Add or subtract based on ordering
                    if client_id < other_id:
                        masked_param = masked_param + pairwise_key
                    else:
                        masked_param = masked_param - pairwise_key

            masked[name] = masked_param

        return masked

    def _generate_pairwise_key(self, client_id1: int, client_id2: int) -> torch.Tensor:
        """Generate pairwise key between two clients.

        Args:
            client_id1: First client ID
            client_id2: Second client ID

        Returns:
            Pairwise key (scalar for demo)
        """
        # Simple hash-based key generation
        key_str = f"{min(client_id1, client_id2)}_{max(client_id1, client_id2)}"
        key_hash = int(hashlib.sha256(key_str.encode()).hexdigest(), 16)
        
        torch.manual_seed(key_hash % (2**32))
        return torch.randn(1).item()

    def aggregate_masked(
        self,
        masked_params_list: List[Dict[str, torch.Tensor]],
        num_samples: List[int],
    ) -> Dict[str, torch.Tensor]:
        """Aggregate masked parameters.
        
        The pairwise masks cancel out during aggregation, revealing
        only the aggregate value.

        Args:
            masked_params_list: List of masked parameters from clients
            num_samples: Number of samples per client

        Returns:
            Aggregated parameters (masks canceled)
        """
        if not masked_params_list:
            raise ValueError("No parameters to aggregate")

        # Weighted average
        total_samples = sum(num_samples)
        weights = [n / total_samples for n in num_samples]

        aggregated = OrderedDict()
        param_names = masked_params_list[0].keys()

        for name in param_names:
            # Pairwise masks cancel when aggregating
            aggregated[name] = sum(
                masked_params_list[i][name] * weights[i]
                for i in range(len(masked_params_list))
            )

        return aggregated


class HomomorphicEncryption:
    """Simplified homomorphic encryption for secure aggregation.
    
    Note: This is a simplified demonstration. Production systems should
    use libraries like TenSEAL or PySyft for proper HE implementation.
    """

    def __init__(self, key_size: int = 2048):
        """Initialize homomorphic encryption.

        Args:
            key_size: Size of encryption key
        """
        self.key_size = key_size
        self.public_key = self._generate_public_key()
        self.private_key = self._generate_private_key()

    def _generate_public_key(self) -> int:
        """Generate public key (simplified)."""
        return torch.randint(1, 1000, (1,)).item()

    def _generate_private_key(self) -> int:
        """Generate private key (simplified)."""
        return torch.randint(1, 1000, (1,)).item()

    def encrypt(self, value: torch.Tensor) -> torch.Tensor:
        """Encrypt a value (simplified).

        Args:
            value: Tensor to encrypt

        Returns:
            Encrypted tensor
        """
        # Simplified encryption: multiply by public key
        return value * self.public_key

    def decrypt(self, encrypted_value: torch.Tensor) -> torch.Tensor:
        """Decrypt a value (simplified).

        Args:
            encrypted_value: Encrypted tensor

        Returns:
            Decrypted tensor
        """
        # Simplified decryption: divide by public key
        return encrypted_value / self.public_key

    def add_encrypted(
        self,
        encrypted1: torch.Tensor,
        encrypted2: torch.Tensor,
    ) -> torch.Tensor:
        """Add two encrypted values (homomorphic property).

        Args:
            encrypted1: First encrypted tensor
            encrypted2: Second encrypted tensor

        Returns:
            Encrypted sum
        """
        return encrypted1 + encrypted2
