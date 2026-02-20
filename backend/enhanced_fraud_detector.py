"""
Jan-Dhan Gateway - Enhanced Fraud Detection with ML
Integrated fraud detection and ledger tamper detection
"""

import numpy as np
from datetime import datetime
from typing import Dict, Optional
import pickle
import os

class EnhancedFraudDetector:
    """
    Enhanced fraud detection with proper ML model integration
    Features:
    1. Transaction fraud detection (Isolation Forest)
    2. Ledger tamper detection (Random Forest)
    3. Feature scaling
    4. Comprehensive analysis
    """
    
    def __init__(self, models_dir: str = '../models'):
        self.models_dir = models_dir
        self.fraud_model = None
        self.fraud_scaler = None
        self.ledger_model = None
        self.is_fraud_model_loaded = False
        self.is_ledger_model_loaded = False
        
        self._load_models()
    
    def _load_models(self):
        """Load all trained ML models"""
        print("🤖 Initializing AI/ML models...")
        
        # Try to load fraud detection model
        fraud_model_path = os.path.join(self.models_dir, 'fraud_model.pkl')
        fraud_scaler_path = os.path.join(self.models_dir, 'fraud_scaler.pkl')
        
        if os.path.exists(fraud_model_path) and os.path.exists(fraud_scaler_path):
            try:
                with open(fraud_model_path, 'rb') as f:
                    self.fraud_model = pickle.load(f)
                
                with open(fraud_scaler_path, 'rb') as f:
                    self.fraud_scaler = pickle.load(f)
                
                self.is_fraud_model_loaded = True
                print("✓ Fraud detection model loaded (ML)")
            except Exception as e:
                print(f"⚠️ Failed to load fraud model: {e}")
                print("  Using rule-based fallback")
        else:
            print("⚠️ Fraud model not found. Using rule-based fallback")
            print(f"  Expected: {fraud_model_path}")
        
        # Try to load ledger tamper model
        ledger_model_path = os.path.join(self.models_dir, 'ledger_tamper_model.pkl')
        
        if os.path.exists(ledger_model_path):
            try:
                with open(ledger_model_path, 'rb') as f:
                    self.ledger_model = pickle.load(f)
                
                self.is_ledger_model_loaded = True
                print("✓ Ledger tamper detection model loaded (ML)")
            except Exception as e:
                print(f"⚠️ Failed to load ledger model: {e}")
        else:
            print("⚠️ Ledger tamper model not found")
            print(f"  Expected: {ledger_model_path}")
    
    def extract_transaction_features(self, transaction_data: Dict) -> np.ndarray:
        """
        Extract 6 features from transaction for ML model
        Returns normalized feature array
        """
        try:
            claim_count = float(transaction_data.get('claim_count', 0))
            days_since_claim = float(transaction_data.get('days_since_claim', 30))
            amount = float(transaction_data.get('amount', 0))
            
            # Time-based features
            now = datetime.now()
            hour = now.hour + now.minute / 60.0  # Decimal hour
            day_of_week = now.weekday()
            
            # Frequency score
            if days_since_claim > 0:
                frequency_score = claim_count / days_since_claim
            else:
                frequency_score = claim_count * 10  # Very suspicious
            
            # Normalize features
            normalized_claims = min(claim_count / 5.0, 1.0)
            normalized_amount = amount / 5000.0
            normalized_hour = hour / 24.0
            normalized_day = day_of_week / 6.0
            
            features = np.array([
                normalized_claims,
                days_since_claim,
                normalized_amount,
                normalized_hour,
                normalized_day,
                frequency_score
            ])
            
            return features.reshape(1, -1)
        
        except Exception as e:
            # Return neutral features on error
            return np.array([[0.5, 30, 0.5, 0.5, 0.5, 0.1]])
    
    def predict_fraud_ml(self, transaction_data: Dict) -> float:
        """
        ML-based fraud prediction using Isolation Forest
        Returns fraud probability (0.0 to 1.0)
        """
        if not self.is_fraud_model_loaded:
            return self._rule_based_fraud_score(transaction_data)
        
        try:
            # Extract and scale features
            features = self.extract_transaction_features(transaction_data)
            features_scaled = self.fraud_scaler.transform(features)
            
            # Get anomaly score
            anomaly_score = self.fraud_model.score_samples(features_scaled)[0]
            
            # Convert to fraud probability (0 = normal, 1 = fraud)
            # Isolation Forest anomaly scores are typically between -0.5 and 0.5
            fraud_prob = max(0.0, min(1.0, 1.0 - (anomaly_score + 0.5)))
            
            return fraud_prob
        
        except Exception as e:
            print(f"ML prediction failed: {e}")
            return self._rule_based_fraud_score(transaction_data)
    
    def _rule_based_fraud_score(self, transaction_data: Dict) -> float:
        """
        Fallback rule-based fraud scoring
        Used when ML model is not available
        """
        score = 0.0
        
        # High claim count
        claim_count = transaction_data.get('claim_count', 0)
        if claim_count >= 5:
            score += 0.4
        elif claim_count >= 4:
            score += 0.3
        elif claim_count >= 3:
            score += 0.2
        
        # Recent claims
        days_since_claim = transaction_data.get('days_since_claim', 30)
        if days_since_claim < 7:
            score += 0.4
        elif days_since_claim < 15:
            score += 0.2
        
        # Off-hours (midnight to 5 AM)
        hour = datetime.now().hour
        if 0 <= hour <= 5:
            score += 0.15
        
        # High frequency
        if claim_count > 2 and days_since_claim < 30:
            score += 0.2
        
        return min(score, 1.0)
    
    def analyze_transaction(self, transaction_data: Dict, threshold: float = 0.75) -> Dict:
        """
        Complete fraud analysis with detailed report
        
        Args:
            transaction_data: Dict with keys:
                - claim_count: Number of claims made
                - days_since_claim: Days since last claim
                - amount: Transaction amount
            threshold: Fraud probability threshold
        
        Returns:
            Analysis dict with fraud probability, risk level, recommendation
        """
        # Get fraud probability
        fraud_prob = self.predict_fraud_ml(transaction_data)
        
        # Classify risk level
        if fraud_prob >= 0.9:
            risk_level = "🔴 CRITICAL"
        elif fraud_prob >= 0.75:
            risk_level = "🟠 HIGH"
        elif fraud_prob >= 0.5:
            risk_level = "🟡 MEDIUM"
        else:
            risk_level = "🟢 LOW"
        
        # Generate recommendation
        if fraud_prob >= 0.9:
            recommendation = "REJECT"
        elif fraud_prob >= 0.75:
            recommendation = "MANUAL_REVIEW"
        else:
            recommendation = "APPROVE"
        
        # Generate explanation
        reasons = []
        claim_count = transaction_data.get('claim_count', 0)
        days_since = transaction_data.get('days_since_claim', 30)
        
        if claim_count > 3:
            reasons.append(f"High claim count ({claim_count})")
        
        if days_since < 15:
            reasons.append(f"Recent claim ({days_since} days ago)")
        
        if fraud_prob >= 0.75 and not reasons:
            reasons.append("Unusual transaction pattern detected by AI")
        
        hour = datetime.now().hour
        if 0 <= hour <= 5:
            reasons.append("Off-hours transaction (midnight-5am)")
        
        return {
            'fraud_probability': round(fraud_prob, 3),
            'is_suspicious': fraud_prob >= threshold,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'reasons': reasons,
            'model_used': 'ML (Isolation Forest)' if self.is_fraud_model_loaded else 'Rule-Based'
        }
    
    def analyze_ledger_entry(self, ledger_data: Dict) -> Dict:
        """
        Analyze ledger entry for tampering
        
        Args:
            ledger_data: Dict with keys:
                - hash_valid: 1 if hash matches, 0 otherwise
                - timestamp_gap: Gap in seconds since last entry
                - checksum_valid: 1 if checksum valid, 0 otherwise
                - field_count: Number of fields in entry
        
        Returns:
            Analysis dict with tamper probability and status
        """
        if not self.is_ledger_model_loaded:
            return self._rule_based_ledger_check(ledger_data)
        
        try:
            # Extract features
            features = np.array([[
                ledger_data.get('hash_valid', 1),
                ledger_data.get('timestamp_gap', 50),
                ledger_data.get('checksum_valid', 1),
                ledger_data.get('field_count', 8)
            ]])
            
            # Predict
            prediction = self.ledger_model.predict(features)[0]
            probabilities = self.ledger_model.predict_proba(features)[0]
            
            tamper_prob = probabilities[1]  # Probability of tampering
            is_tampered = prediction == 1
            
            # Generate reasons
            reasons = []
            if ledger_data.get('hash_valid', 1) == 0:
                reasons.append("Hash mismatch detected")
            
            if ledger_data.get('timestamp_gap', 0) > 1000:
                reasons.append(f"Unusual timestamp gap ({ledger_data.get('timestamp_gap')} seconds)")
            
            if ledger_data.get('checksum_valid', 1) == 0:
                reasons.append("Checksum validation failed")
            
            if ledger_data.get('field_count', 8) < 8:
                reasons.append(f"Missing fields (only {ledger_data.get('field_count')} of 8)")
            
            return {
                'is_tampered': is_tampered,
                'tamper_probability': round(tamper_prob, 3),
                'status': '🚨 TAMPERED' if is_tampered else '✅ VALID',
                'reasons': reasons,
                'model_used': 'ML (Random Forest)'
            }
        
        except Exception as e:
            print(f"Ledger analysis failed: {e}")
            return self._rule_based_ledger_check(ledger_data)
    
    def _rule_based_ledger_check(self, ledger_data: Dict) -> Dict:
        """Rule-based ledger tampering check"""
        is_tampered = False
        reasons = []
        
        if ledger_data.get('hash_valid', 1) == 0:
            is_tampered = True
            reasons.append("Hash mismatch detected")
        
        if ledger_data.get('timestamp_gap', 0) > 1000:
            is_tampered = True
            reasons.append("Unusual timestamp gap")
        
        if ledger_data.get('checksum_valid', 1) == 0:
            is_tampered = True
            reasons.append("Checksum validation failed")
        
        return {
            'is_tampered': is_tampered,
            'tamper_probability': 1.0 if is_tampered else 0.0,
            'status': '🚨 TAMPERED' if is_tampered else '✅ VALID',
            'reasons': reasons,
            'model_used': 'Rule-Based'
        }


# Legacy compatibility wrapper
class FraudDetector(EnhancedFraudDetector):
    """Wrapper for backward compatibility with existing code"""
    pass
