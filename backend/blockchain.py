"""
Jan-Dhan Gateway - Blockchain Ledger System
Implements immutable hash-linked ledger with Merkle tree batching
"""

import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from config import Config
from crypto_utils import CryptoUtils

class BlockchainLedger:
    """Immutable hash-chain ledger for transaction recording"""

    def __init__(self, ledger_path: str = None):
        self.ledger_path = ledger_path or Config.LEDGER_PATH
        self.last_hash = Config.GENESIS_HASH
        self._initialize_ledger()
        self._load_last_hash()

    def _initialize_ledger(self):
        """Create ledger file if it doesn't exist"""
        Config.ensure_data_directory()
        if not os.path.exists(self.ledger_path):
            with open(self.ledger_path, 'w') as f:
                f.write("# Jan-Dhan Gateway Immutable Ledger\n")
                f.write(f"# Genesis Hash: {Config.GENESIS_HASH}\n")
                f.write("# Format: Timestamp|Citizen_Hash|Scheme|Amount|Previous_Hash|Current_Hash\n")
                f.write("\n")

    def _load_last_hash(self):
        """Load the hash of the last transaction in chain"""
        try:
            with open(self.ledger_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                if lines:
                    last_line = lines[-1]
                    parts = last_line.split('|')
                    if len(parts) >= 6:
                        self.last_hash = parts[5]  # Current hash becomes previous hash for next
        except Exception:
            self.last_hash = Config.GENESIS_HASH

    def append_transaction(self, citizen_id_hash: str, scheme: str, amount: float) -> Dict:
        """
        Append approved transaction to immutable ledger

        Args:
            citizen_id_hash: SHA-256 hash of citizen ID
            scheme: Benefit scheme name
            amount: Transaction amount

        Returns:
            Dictionary with transaction details and hash
        """
        timestamp = Config.get_current_timestamp()

        # Compute current transaction hash
        current_hash = CryptoUtils.hash_transaction(
            timestamp, citizen_id_hash, scheme, amount, self.last_hash
        )

        # Format ledger entry
        entry = f"{timestamp}|{citizen_id_hash}|{scheme}|{amount}|{self.last_hash}|{current_hash}\n"

        # Append to file (immutable operation)
        with open(self.ledger_path, 'a') as f:
            f.write(entry)

        # Update last hash for next transaction
        self.last_hash = current_hash

        return {
            'timestamp': timestamp,
            'citizen_hash': citizen_id_hash,
            'scheme': scheme,
            'amount': amount,
            'previous_hash': self.last_hash,
            'transaction_hash': current_hash
        }

    def verify_integrity(self) -> Tuple[bool, Optional[int], str]:
        """
        Verify complete integrity of hash-chain ledger

        Returns:
            Tuple of (is_valid, corrupted_line, integrity_hash)
        """
        try:
            with open(self.ledger_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

            is_valid, corrupted_line = CryptoUtils.verify_hash_chain(lines)

            # Compute file integrity hash
            integrity_hash = CryptoUtils.compute_file_integrity_hash(self.ledger_path)

            return is_valid, corrupted_line, integrity_hash

        except Exception as e:
            return False, None, ""

    def get_all_transactions(self) -> List[Dict]:
        """
        Retrieve all transactions from ledger

        Returns:
            List of transaction dictionaries
        """
        transactions = []

        try:
            with open(self.ledger_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split('|')
                        if len(parts) >= 6:
                            transactions.append({
                                'timestamp': parts[0],
                                'citizen_hash': parts[1],
                                'scheme': parts[2],
                                'amount': float(parts[3]),
                                'previous_hash': parts[4],
                                'current_hash': parts[5]
                            })
        except Exception:
            pass

        return transactions

    def get_transaction_count(self) -> int:
        """Get total number of transactions in ledger"""
        return len(self.get_all_transactions())

    def get_total_disbursed(self) -> float:
        """Calculate total amount disbursed from ledger"""
        transactions = self.get_all_transactions()
        return sum(tx['amount'] for tx in transactions)

    def check_duplicate_claim(self, citizen_id_hash: str) -> bool:
        """
        Check if citizen ID hash already exists in ledger (replay attack detection)

        Args:
            citizen_id_hash: Hashed citizen ID to check

        Returns:
            True if duplicate found, False otherwise
        """
        transactions = self.get_all_transactions()
        return any(tx['citizen_hash'] == citizen_id_hash for tx in transactions)

    def compute_merkle_root(self) -> str:
        """
        Compute Merkle root of all transaction hashes for batch verification
        UNIQUE FEATURE: Advanced cryptographic proof

        Returns:
            Merkle root hash
        """
        transactions = self.get_all_transactions()
        hashes = [tx['current_hash'] for tx in transactions]
        return CryptoUtils.build_merkle_tree(hashes)

    def get_ledger_statistics(self) -> Dict:
        """
        Get comprehensive ledger statistics

        Returns:
            Dictionary with ledger stats
        """
        is_valid, corrupted_line, integrity_hash = self.verify_integrity()
        transactions = self.get_all_transactions()

        # Calculate scheme-wise distribution
        scheme_stats = {}
        for tx in transactions:
            scheme = tx['scheme']
            if scheme not in scheme_stats:
                scheme_stats[scheme] = {'count': 0, 'amount': 0}
            scheme_stats[scheme]['count'] += 1
            scheme_stats[scheme]['amount'] += tx['amount']

        return {
            'total_transactions': len(transactions),
            'total_disbursed': sum(tx['amount'] for tx in transactions),
            'is_valid': is_valid,
            'corrupted_line': corrupted_line,
            'integrity_hash': integrity_hash,
            'merkle_root': self.compute_merkle_root(),
            'scheme_statistics': scheme_stats,
            'last_transaction_hash': self.last_hash
        }
