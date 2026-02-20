"""
Jan-Dhan Gateway - Cryptographic Utilities
Implements SHA-256 hashing, Merkle tree verification, and hash-chain validation
"""

import hashlib
import json
from typing import List, Tuple, Optional

class CryptoUtils:
    """Advanced cryptographic utilities for secure transaction processing"""

    @staticmethod
    def hash_citizen_id(citizen_id: str) -> str:
        """
        Convert 12-digit Citizen_ID into SHA-256 hash

        Args:
            citizen_id: 12-digit citizen identifier

        Returns:
            64-character hexadecimal hash
        """
        return hashlib.sha256(str(citizen_id).encode()).hexdigest()

    @staticmethod
    def hash_transaction(timestamp: str, citizen_hash: str, scheme: str, 
                        amount: float, previous_hash: str) -> str:
        """
        Create hash-chain link for transaction
        Formula: SHA256(Timestamp + CitizenHash + Scheme + Amount + PreviousHash)

        Args:
            timestamp: Transaction timestamp
            citizen_hash: Hashed citizen ID
            scheme: Benefit scheme name
            amount: Transaction amount
            previous_hash: Hash of previous transaction

        Returns:
            Transaction hash for blockchain ledger
        """
        data = f"{timestamp}{citizen_hash}{scheme}{amount}{previous_hash}"
        return hashlib.sha256(data.encode()).hexdigest()

    @staticmethod
    def compute_file_integrity_hash(file_path: str) -> str:
        """
        Compute SHA-256 hash of entire ledger file for tamper detection

        Args:
            file_path: Path to ledger file

        Returns:
            File integrity hash
        """
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256()
                # Read file in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    file_hash.update(chunk)
                return file_hash.hexdigest()
        except FileNotFoundError:
            return hashlib.sha256(b"").hexdigest()  # Empty file hash

    @staticmethod
    def build_merkle_tree(hashes: List[str]) -> str:
        """
        Build Merkle tree root from transaction hashes for batch verification
        This is a UNIQUE FEATURE for advanced cryptographic proof

        Args:
            hashes: List of transaction hashes

        Returns:
            Merkle root hash
        """
        if not hashes:
            return hashlib.sha256(b"empty").hexdigest()

        if len(hashes) == 1:
            return hashes[0]

        # Build tree level by level
        while len(hashes) > 1:
            if len(hashes) % 2 != 0:
                hashes.append(hashes[-1])  # Duplicate last hash if odd

            next_level = []
            for i in range(0, len(hashes), 2):
                combined = hashes[i] + hashes[i + 1]
                next_level.append(hashlib.sha256(combined.encode()).hexdigest())

            hashes = next_level

        return hashes[0]

    @staticmethod
    def verify_merkle_proof(transaction_hash: str, proof: List[Tuple[str, str]], 
                           root: str) -> bool:
        """
        Verify a transaction is part of Merkle tree

        Args:
            transaction_hash: Hash to verify
            proof: List of (hash, position) tuples for verification path
            root: Expected Merkle root

        Returns:
            True if verification succeeds
        """
        current_hash = transaction_hash

        for sibling_hash, position in proof:
            if position == 'left':
                combined = sibling_hash + current_hash
            else:
                combined = current_hash + sibling_hash

            current_hash = hashlib.sha256(combined.encode()).hexdigest()

        return current_hash == root

    @staticmethod
    def generate_transaction_signature(transaction_data: dict) -> str:
        """
        Generate unique signature for transaction (prevents replay attacks)

        Args:
            transaction_data: Dictionary containing transaction details

        Returns:
            Unique transaction signature
        """
        data_str = json.dumps(transaction_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    @staticmethod
    def verify_hash_chain(ledger_lines: List[str]) -> Tuple[bool, Optional[int]]:
        """
        Verify integrity of entire hash-chain ledger

        Args:
            ledger_lines: List of ledger entries

        Returns:
            Tuple of (is_valid, corrupted_line_number)
        """
        if not ledger_lines:
            return True, None

        previous_hash = "0" * 64  # Genesis hash

        for idx, line in enumerate(ledger_lines):
            try:
                parts = line.strip().split('|')
                if len(parts) < 6:
                    continue

                timestamp, citizen_hash, scheme, amount, prev_hash, current_hash = parts

                # Verify previous hash matches
                if prev_hash != previous_hash:
                    return False, idx + 1

                # Recompute and verify current hash
                computed_hash = CryptoUtils.hash_transaction(
                    timestamp, citizen_hash, scheme, float(amount), prev_hash
                )

                if computed_hash != current_hash:
                    return False, idx + 1

                previous_hash = current_hash

            except Exception:
                return False, idx + 1

        return True, None

# Export utility functions for easy import
hash_id = CryptoUtils.hash_citizen_id
hash_tx = CryptoUtils.hash_transaction
verify_chain = CryptoUtils.verify_hash_chain
merkle_root = CryptoUtils.build_merkle_tree
