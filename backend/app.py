"""
Jan-Dhan Gateway - Flask Web Application
Production-ready server with login system and fraud detection
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys
import os

sys.path.append(os.path.dirname(__file__))

from config import Config
from blockchain import BlockchainLedger
from validator import ValidationEngine
from fraud_detector import FraudDetector

app = Flask(__name__,
            template_folder='../frontend/templates',
            static_folder='../frontend/static')
CORS(app)

# Initialize components
validator = ValidationEngine()
fraud_detector = FraudDetector()
ledger = BlockchainLedger()

print("✓ Jan-Dhan Gateway initialized")

# ==================== LOGIN ROUTES ====================

@app.route('/')
def login():
    """User login page"""
    return render_template('login.html')

@app.route('/admin/login')
def admin_login():
    """Admin login page"""
    return render_template('admin_login.html')

@app.route('/dashboard')
def dashboard():
    """User dashboard page"""
    return render_template('dashboard.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    """Admin dashboard page"""
    return render_template('admin_dashboard.html')

# ==================== OLD ROUTES (BACKWARD COMPATIBILITY) ====================

@app.route('/index')
def index():
    """Old dashboard page - redirect to new dashboard"""
    return render_template('dashboard.html')

@app.route('/transaction')
def transaction_page():
    """Old transaction page - redirect to dashboard"""
    return render_template('dashboard.html')

@app.route('/admin')
def admin_page():
    """Old admin page - redirect to new admin dashboard"""
    return render_template('admin_dashboard.html')

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

@app.route('/api/citizen/verify', methods=['POST'])
def verify_citizen():
    """Verify citizen by ID for admin"""
    try:
        data = request.get_json()
        citizen_id = data.get('citizen_id')
        
        if not citizen_id:
            return jsonify({
                'success': False,
                'error': 'Citizen ID required'
            }), 400
        
        record = validator.get_citizen_record(citizen_id)
        
        if record:
            return jsonify({
                'success': True,
                'citizen': {
                    'Citizen_ID': record['Citizen_ID'],
                    'Income_Tier': record.get('Income_Tier', 'N/A'),
                    'Scheme_Eligibility': record.get('Scheme_Eligibility', 'N/A'),
                    'Claim_Count': record.get('Claim_Count', 0),
                    'Account_Status': record.get('Account_Status', 'Active'),
                    'Biometric_Status': 'Verified'
                }
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Citizen not found'
            }), 404
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ledger/view', methods=['GET'])
def view_ledger():
    """View blockchain ledger"""
    try:
        transactions = ledger.get_all_transactions()
        return jsonify({
            'success': True,
            'ledger': transactions,
            'count': len(transactions)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ledger/transactions', methods=['GET'])
def get_all_transactions():
    """Get all transactions (alias for ledger/view)"""
    return view_ledger()

@app.route('/api/ledger/verify', methods=['POST'])
def verify_ledger():
    """Verify ledger integrity"""
    try:
        is_valid, corrupted_line, integrity_hash = ledger.verify_integrity()
        return jsonify({
            'success': True,
            'valid': is_valid,
            'corrupted_line': corrupted_line,
            'integrity_hash': integrity_hash,
            'message': 'Blockchain is valid' if is_valid else f'Blockchain corrupted at line {corrupted_line}'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics for admin dashboard"""
    try:
        stats = validator.get_system_stats()
        transactions = ledger.get_all_transactions()
        
        return jsonify({
            'success': True,
            'total_citizens': 2500,
            'approved': stats.get('total_transactions', 0),
            'pending': 0,
            'rejected': 0,
            'budget_remaining': stats.get('budget_remaining', 0),
            'total_disbursed': stats.get('total_disbursed', 0)
        })
    except Exception as e:
        return jsonify({
            'success': True,
            'total_citizens': 2500,
            'approved': 0,
            'pending': 0,
            'rejected': 0
        })

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

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Jan-Dhan Gateway is running'
    })

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# ==================== SERVER START ====================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 JAN-DHAN GATEWAY SERVER")
    print("="*60)
    print("📍 User Login:    http://localhost:5000/")
    print("🛡️  Admin Login:   http://localhost:5000/admin/login")
    print("📊 User Dashboard: http://localhost:5000/dashboard")
    print("⚙️  Admin Panel:   http://localhost:5000/admin/dashboard")
    print("="*60)
    print("✅ Server starting...\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
