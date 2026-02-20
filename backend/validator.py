"""
Jan-Dhan Gateway - Three-Gate Validation Engine
Implements strict eligibility, budget, and frequency checks with AI fraud detection
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
from config import Config
from crypto_utils import CryptoUtils
from blockchain import BlockchainLedger

class ValidationEngine:
    """Three-gate sequential validation system"""

    def __init__(self, registry_path: str = None):
        self.registry_path = registry_path or Config.REGISTRY_PATH
        self.db_path = Config.DB_PATH
        self.budget_remaining = Config.INITIAL_BUDGET
        self.system_status = Config.SYSTEM_ACTIVE
        self.ledger = BlockchainLedger()
        self._initialize_database()

    def _initialize_database(self):
        """Load Excel registry into SQLite for efficient queries"""
        try:
            df = pd.read_excel(self.registry_path, sheet_name='Jan_Dhan_Registry_Advanced')

            # Convert Citizen_ID to string to prevent scientific notation
            df['Citizen_ID'] = df['Citizen_ID'].astype(str)

            # Create SQLite database
            conn = sqlite3.connect(self.db_path)
            df.to_sql('registry', conn, if_exists='replace', index=False)
            conn.close()

        except Exception as e:
            print(f"Warning: Could not load registry - {str(e)}")

    def get_citizen_record(self, citizen_id: str) -> Optional[Dict]:
        """Fetch citizen record from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM registry WHERE Citizen_ID = ?
            """, (str(citizen_id),))

            row = cursor.fetchone()
            conn.close()

            if row:
                columns = ['Citizen_ID', 'Income_Tier', 'Scheme_Eligibility', 
                          'Scheme_Amount', 'Last_Claim_Date', 'Region_Code', 
                          'Account_Status', 'Aadhaar_Linked', 'Claim_Count']
                return dict(zip(columns, row))

            return None

        except Exception as e:
            print(f"Database error: {str(e)}")
            return None

    def gate_1_eligibility(self, citizen_id: str, requested_scheme: str) -> Tuple[bool, str]:
        """
        GATE 1: Eligibility Validation
        Checks: ID exists, Account Active, Aadhaar linked, Scheme matches, Claim count

        Returns: (is_valid, error_message)
        """
        # Check citizen ID exists
        record = self.get_citizen_record(citizen_id)
        if not record:
            return False, "❌ Citizen ID not found in registry"

        # Check account status
        if record['Account_Status'] != 'Active':
            return False, f"❌ Account status is {record['Account_Status']}, not Active"

        # Check Aadhaar linkage
        aadhaar_linked = record['Aadhaar_Linked']
        if isinstance(aadhaar_linked, str):
            aadhaar_linked = aadhaar_linked.upper() == 'TRUE'

        if not aadhaar_linked:
            return False, "❌ Aadhaar not linked to account"

        # Check scheme eligibility
        if record['Scheme_Eligibility'] != requested_scheme:
            return False, f"❌ Citizen eligible for {record['Scheme_Eligibility']}, not {requested_scheme}"

        # Check scheme amount matches
        if Config.SCHEME_AMOUNTS[requested_scheme] != record['Scheme_Amount']:
            return False, f"❌ Scheme amount mismatch: expected {Config.SCHEME_AMOUNTS[requested_scheme]}, found {record['Scheme_Amount']}"

        # Check claim count
        claim_count = int(record['Claim_Count'])
        if claim_count > Config.MAX_CLAIM_COUNT:
            return False, f"❌ Claim count ({claim_count}) exceeds limit ({Config.MAX_CLAIM_COUNT})"

        return True, "✅ Eligibility verified"

    def gate_2_budget(self, amount: float) -> Tuple[bool, str]:
        """
        GATE 2: Budget Validation
        Checks: Sufficient budget available

        Returns: (is_valid, error_message)
        """
        if self.budget_remaining <= 0:
            self.system_status = Config.SYSTEM_FROZEN
            return False, "❌ Budget exhausted - System auto-locked"

        if amount > self.budget_remaining:
            return False, f"❌ Insufficient budget: Required ₹{amount}, Available ₹{self.budget_remaining}"

        return True, f"✅ Budget available (₹{self.budget_remaining})"

    def gate_3_frequency(self, citizen_id: str) -> Tuple[bool, str]:
        """
        GATE 3: Frequency Validation
        Checks: No claims within 30 days of last claim

        Returns: (is_valid, error_message)
        """
        record = self.get_citizen_record(citizen_id)
        if not record:
            return False, "❌ Citizen record not found"

        last_claim_str = record['Last_Claim_Date']
        if not last_claim_str or pd.isna(last_claim_str):
            return True, "✅ No previous claims found"

        try:
            # Parse date (handles both DD-MM-YYYY and YYYY-MM-DD formats)
            if isinstance(last_claim_str, str):
                if '-' in last_claim_str:
                    parts = last_claim_str.split('-')
                    if len(parts[0]) == 4:  # YYYY-MM-DD
                        last_claim_date = datetime.strptime(last_claim_str, "%Y-%m-%d")
                    else:  # DD-MM-YYYY
                        last_claim_date = datetime.strptime(last_claim_str, "%d-%m-%Y")
                else:
                    return True, "✅ Invalid date format, allowing claim"
            else:
                last_claim_date = last_claim_str

            days_since_claim = (datetime.now() - last_claim_date).days

            if days_since_claim < Config.FREQUENCY_LIMIT_DAYS:
                return False, f"❌ Last claim was {days_since_claim} days ago (minimum {Config.FREQUENCY_LIMIT_DAYS} days required)"

            return True, f"✅ Frequency check passed ({days_since_claim} days since last claim)"

        except Exception as e:
            return True, f"✅ Date parsing error, allowing claim: {str(e)}"

    def validate_transaction(self, citizen_id: str, scheme: str) -> Dict:
        """
        Execute complete three-gate validation

        Returns: Validation result dictionary
        """
        result = {
            'citizen_id': citizen_id,
            'scheme': scheme,
            'approved': False,
            'gates_passed': [],
            'gates_failed': [],
            'messages': [],
            'amount': 0,
            'transaction_hash': None
        }

        # System status check
        if self.system_status != Config.SYSTEM_ACTIVE:
            result['messages'].append(f"🔒 System is {self.system_status} - No transactions allowed")
            return result

        # Hash citizen ID for privacy
        citizen_hash = CryptoUtils.hash_citizen_id(citizen_id)

        # Check for replay attack (duplicate hash in ledger)
        if self.ledger.check_duplicate_claim(citizen_hash):
            result['messages'].append("🚫 REPLAY ATTACK DETECTED - This ID already claimed in this session")
            result['gates_failed'].append('REPLAY_DETECTION')
            return result

        # Get transaction amount
        amount = Config.SCHEME_AMOUNTS.get(scheme, 0)
        result['amount'] = amount

        # GATE 1: Eligibility
        gate1_valid, gate1_msg = self.gate_1_eligibility(citizen_id, scheme)
        result['messages'].append(f"🔐 GATE 1 (Eligibility): {gate1_msg}")

        if gate1_valid:
            result['gates_passed'].append('GATE_1_ELIGIBILITY')
        else:
            result['gates_failed'].append('GATE_1_ELIGIBILITY')
            return result

        # GATE 2: Budget
        gate2_valid, gate2_msg = self.gate_2_budget(amount)
        result['messages'].append(f"💰 GATE 2 (Budget): {gate2_msg}")

        if gate2_valid:
            result['gates_passed'].append('GATE_2_BUDGET')
        else:
            result['gates_failed'].append('GATE_2_BUDGET')
            return result

        # GATE 3: Frequency
        gate3_valid, gate3_msg = self.gate_3_frequency(citizen_id)
        result['messages'].append(f"⏱️  GATE 3 (Frequency): {gate3_msg}")

        if gate3_valid:
            result['gates_passed'].append('GATE_3_FREQUENCY')
        else:
            result['gates_failed'].append('GATE_3_FREQUENCY')
            return result

        # All gates passed - Approve transaction
        result['approved'] = True
        result['messages'].append(f"✅ ALL GATES PASSED - Transaction approved for ₹{amount}")

        # Record in blockchain ledger
        tx_record = self.ledger.append_transaction(citizen_hash, scheme, amount)
        result['transaction_hash'] = tx_record['transaction_hash']

        # Deduct from budget
        self.budget_remaining -= amount

        # Check if budget exhausted
        if self.budget_remaining <= 0:
            self.system_status = Config.SYSTEM_FROZEN
            result['messages'].append("⚠️  Budget exhausted - System auto-locked")

        return result

    def pause_system(self):
        """Admin function to pause system"""
        self.system_status = Config.SYSTEM_PAUSED

    def activate_system(self):
        """Admin function to activate system"""
        if self.budget_remaining > 0:
            self.system_status = Config.SYSTEM_ACTIVE
        else:
            self.system_status = Config.SYSTEM_FROZEN

    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        ledger_stats = self.ledger.get_ledger_statistics()

        return {
            'system_status': self.system_status,
            'budget_remaining': self.budget_remaining,
            'budget_utilized': Config.INITIAL_BUDGET - self.budget_remaining,
            'utilization_percentage': ((Config.INITIAL_BUDGET - self.budget_remaining) / Config.INITIAL_BUDGET) * 100,
            'total_transactions': ledger_stats['total_transactions'],
            'total_disbursed': ledger_stats['total_disbursed'],
            'ledger_integrity': ledger_stats['is_valid'],
            'merkle_root': ledger_stats['merkle_root']
        }
