"""
Jan-Dhan Gateway - Configuration Settings
Author: DEV-HOURS Hackathon 2026
"""

import os
from datetime import datetime

class Config:
    # System Configuration
    INITIAL_BUDGET = 1000000  # ₹10,00,000 INR
    MAX_CLAIM_COUNT = 3
    FREQUENCY_LIMIT_DAYS = 30
    
    # System States
    SYSTEM_ACTIVE = "Active"
    SYSTEM_PAUSED = "Paused"
    SYSTEM_FROZEN = "Frozen"
    
    # File Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
    REGISTRY_PATH = os.path.join(DATA_DIR, 'jan_dhan_registry_advanced.xlsx')
    LEDGER_PATH = os.path.join(DATA_DIR, 'ledger.txt')
    DB_PATH = os.path.join(DATA_DIR, 'registry.db')
    MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'fraud_model.pkl')
    
    # Blockchain Configuration
    GENESIS_HASH = "0" * 64  # Initial hash for blockchain
    HASH_ALGORITHM = "SHA256"
    
    # Security
    SECRET_KEY = os.urandom(24).hex()
    ADMIN_PASSWORD_HASH = "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8"  # "password"
    
    # Scheme Mapping (Fixed amounts)
    SCHEME_AMOUNTS = {
        'Pension': 3000,
        'Health': 5000,
        'Food': 2000
    }  # ← FIXED: Added closing brace
    
    # ML Configuration
    FRAUD_THRESHOLD = 0.75  # Probability threshold for flagging suspicious activity
    
    # UI Configuration
    #RECORDS_PER_PAGE = 50
    #DASHBOARD_REFRESH_INTERVAL = 5000  # milliseconds
    
    @staticmethod
    def get_current_timestamp():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def ensure_data_directory():
        """Create data directory if it doesn't exist"""
        os.makedirs(Config.DATA_DIR, exist_ok=True)