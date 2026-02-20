"""
Jan-Dhan Gateway - Flask Web Application
REST API server for transaction processing
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys
import os

sys.path.append(os.path.dirname(__file__))

from config import Config
from blockchain import BlockchainLedger
from validator import ValidationEngine
#from fraud_detector import FraudDetector
from enhanced_fraud_detector import EnhancedFraudDetector as FraudDetector

app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')
CORS(app)

# Initialize components
validator = ValidationEngine()
fraud_detector = FraudDetector()
ledger = BlockchainLedger()

# ==================== HTML PAGE ROUTES ====================

@app.route('/')
def index():
    """Dashboard page"""
    return render_template('index.html')

@app.route('/transaction')
def transaction_page():
    """Transaction submission page"""
    return render_template('transaction.html')

@app.route('/admin')
def admin_page():
    """Admin control panel page"""
    return render_template('admin.html')

# ==================== API ENDPOINTS ====================

@app.route('/api/system/status', methods=['GET'])
def get_system_status():
    """Get current system status and statistics"""
    try:
        stats = validator.get_system_stats()
        is_valid, corrupted_line, integrity_hash = ledger.verify_integrity()
        
        return jsonify({
            'success': True,
            'system_status': stats['system_status'],
            'budget': {
                'remaining': stats['budget_remaining'],
                'utilized': stats['budget_utilized'],
                'percentage': round(stats['utilization_percentage'], 2)
            },
            'transactions': {
                'total': stats['total_transactions'],
                'total_disbursed': stats['total_disbursed']
            },
            'ledger': {
                'is_valid': is_valid,
                'merkle_root': stats['merkle_root'][:16] + '...'
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/transaction/submit', methods=['POST'])
def submit_transaction():
    """Submit and validate transaction through three-gate system"""
    try:
        data = request.get_json()
        citizen_id = data.get('citizen_id')
        scheme = data.get('scheme')
        
        if not citizen_id or not scheme:
            return jsonify({'success': False, 'error': 'Missing citizen_id or scheme'}), 400
        
        # Validate through three-gate system
        result = validator.validate_transaction(citizen_id, scheme)
        
        # Add fraud detection analysis if transaction approved
        if result['approved']:
            # Get citizen record for fraud analysis
            record = validator.get_citizen_record(citizen_id)
            if record:
                fraud_data = {
                    'claim_count': int(record['Claim_Count']),
                    'days_since_claim': 30,  # Default value
                    'amount': result['amount']
                }
                
                fraud_analysis = fraud_detector.analyze_transaction(fraud_data)
                result['fraud_analysis'] = fraud_analysis
        
        return jsonify({
            'success': True,
            'transaction': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/pause', methods=['POST'])
def pause_system():
    """Pause transaction processing (admin function)"""
    validator.pause_system()
    return jsonify({'success': True, 'status': validator.system_status})

@app.route('/api/admin/activate', methods=['POST'])
def activate_system():
    """Activate transaction processing (admin function)"""
    validator.activate_system()
    return jsonify({'success': True, 'status': validator.system_status})

@app.route('/api/ledger/transactions', methods=['GET'])
def get_all_transactions():
    """Get all transactions from blockchain ledger"""
    try:
        transactions = ledger.get_all_transactions()
        return jsonify({
            'success': True,
            'transactions': transactions,
            'count': len(transactions)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== SERVER START ====================

if __name__ == '__main__':
    print("🚀 JAN-DHAN GATEWAY SERVER")
    print("=" * 50)
    print("🌐 Dashboard:    http://localhost:5000")
    print("💳 Transaction:  http://localhost:5000/transaction")
    print("⚙️  Admin Panel:  http://localhost:5000/admin")
    print("=" * 50)
    print("✅ Server starting...")
    
    app.run(debug=True, host='0.0.0.0', port=5000)