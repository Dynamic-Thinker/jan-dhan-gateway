"""
Jan-Dhan Gateway - Production Fraud Detector
Uses properly calibrated ML models with real-world performance
"""

import numpy as np
from datetime import datetime
from typing import Dict
import pickle
import os

class ProductionFraudDetector:
    """
    Production-grade fraud detection with calibrated models
    """
    
    def __init__(self, models_dir: str = '../models'):
        self.models_dir = models_dir
        self.fraud_model = None
        self.fraud_scaler = None
        self.min_score = None
        self.max_score = None
        self.features = None
        self.supervised_model = None
        self.supervised_scaler = None
        self.ledger_model = None
        self.is_fraud_model_loaded = False
        self.is_supervised_loaded = False
        self.is_ledger_model_loaded = False
        
        self._load_models()
    
    def _load_models(self):
        """Load all calibrated models"""
        print("🤖 Loading production ML models...")
        
        # Try to load calibrated Isolation Forest
        fraud_model_path = os.path.join(self.models_dir, 'fraud_model_calibrated.pkl')
        
        if os.path.exists(fraud_model_path):
            try:
                with open(fraud_model_path, 'rb') as f:
                    fraud_data = pickle.load(f)
                    self.fraud_model = fraud_data['model']
                    self.fraud_scaler = fraud_data['scaler']
                    self.min_score = fraud_data['min_score']
                    self.max_score = fraud_data['max_score']
                    self.features = fraud_data['features']
                
                self.is_fraud_model_loaded = True
                print("✓ Calibrated fraud detection model loaded")
            except Exception as e:
                print(f"⚠️  Failed to load fraud model: {e}")
        else:
            print("⚠️  Calibrated model not found, using rule-based fallback")
        
        # Try to load supervised model
        supervised_path = os.path.join(self.models_dir, 'fraud_model_supervised.pkl')
        if os.path.exists(supervised_path):
            try:
                with open(supervised_path, 'rb') as f:
                    supervised_data = pickle.load(f)
                    self.supervised_model = supervised_data['model']
                    self.supervised_scaler = supervised_data['scaler']
                
                self.is_supervised_loaded = True
                print("✓ Supervised fraud model loaded")
            except Exception as e:
                print(f"⚠️  Failed to load supervised model: {e}")
        
        # Try to load ledger model
        ledger_path = os.path.join(self.models_dir, 'ledger_tamper_model_calibrated.pkl')
        if os.path.exists(ledger_path):
            try:
                with open(ledger_path, 'rb') as f:
                    self.ledger_model = pickle.load(f)
                
                self.is_ledger_model_loaded = True
                print("✓ Ledger tamper model loaded")
            except Exception as e:
                print(f"⚠️  Failed to load ledger model: {e}")
    
    def extract_features(self, transaction_data: Dict) -> np.ndarray:
        """Extract features for ML model"""
        claim_count = float(transaction_data.get('claim_count', 0))
        days_since_claim = float(transaction_data.get('days_since_claim', 30))
        amount = float(transaction_data.get('amount', 0))
        
        # Time features
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()
        
        # Frequency score
        if days_since_claim > 0:
            frequency_score = claim_count / days_since_claim
        else:
            frequency_score = claim_count * 10
        
        # Additional features
        is_suspended = transaction_data.get('is_suspended', 0)
        is_blacklisted = transaction_data.get('is_blacklisted', 0)
        
        features = [
            claim_count,
            days_since_claim,
            amount,
            hour,
            day_of_week,
            frequency_score,
            is_suspended,
            is_blacklisted
        ]
        
        return np.array([features])
    
    def predict_fraud_probability(self, transaction_data: Dict) -> float:
        """Predict fraud probability with calibrated model"""
        if not self.is_fraud_model_loaded:
            return self._rule_based_fraud_score(transaction_data)
        
        try:
            features = self.extract_features(transaction_data)
            features_scaled = self.fraud_scaler.transform(features)
            
            # Get anomaly score
            anomaly_score = self.fraud_model.score_samples(features_scaled)[0]
            
            # Calibrated probability calculation
            normalized_score = (anomaly_score - self.min_score) / (self.max_score - self.min_score)
            fraud_prob = 1 - normalized_score
            fraud_prob = max(0.0, min(1.0, fraud_prob))
            
            return fraud_prob
        
        except Exception as e:
            print(f"ML prediction failed: {e}")
            return self._rule_based_fraud_score(transaction_data)
    
    def _rule_based_fraud_score(self, transaction_data: Dict) -> float:
        """Fallback rule-based scoring"""
        score = 0.0
        
        claim_count = transaction_data.get('claim_count', 0)
        days_since_claim = transaction_data.get('days_since_claim', 30)
        is_suspended = transaction_data.get('is_suspended', 0)
        is_blacklisted = transaction_data.get('is_blacklisted', 0)
        
        # Account status
        if is_blacklisted:
            score += 0.8
        elif is_suspended:
            score += 0.5
        
        # Claim frequency
        if claim_count >= 5:
            score += 0.4
        elif claim_count >= 4:
            score += 0.3
        
        # Recency
        if days_since_claim < 7:
            score += 0.3
        elif days_since_claim < 15:
            score += 0.15
        
        # Off-hours
        hour = datetime.now().hour
        if 0 <= hour <= 5:
            score += 0.15
        
        return min(score, 1.0)
    
    def analyze_transaction(self, transaction_data: Dict, threshold: float = 0.75) -> Dict:
        """
        Complete fraud analysis
        
        Args:
            transaction_data: Dict with:
                - claim_count: Number of claims
                - days_since_claim: Days since last claim
                - amount: Transaction amount
                - is_suspended: 0 or 1
                - is_blacklisted: 0 or 1
        
        Returns:
            Analysis with fraud probability, risk level, recommendation
        """
        fraud_prob = self.predict_fraud_probability(transaction_data)
        
        # Classify risk
        if fraud_prob >= 0.9:
            risk_level = "🔴 CRITICAL"
            recommendation = "REJECT"
        elif fraud_prob >= 0.75:
            risk_level = "🟠 HIGH"
            recommendation = "MANUAL_REVIEW"
        elif fraud_prob >= 0.5:
            risk_level = "🟡 MEDIUM"
            recommendation = "APPROVE with monitoring"
        else:
            risk_level = "🟢 LOW"
            recommendation = "APPROVE"
        
        # Generate reasons
        reasons = []
        claim_count = transaction_data.get('claim_count', 0)
        days_since = transaction_data.get('days_since_claim', 30)
        is_suspended = transaction_data.get('is_suspended', 0)
        is_blacklisted = transaction_data.get('is_blacklisted', 0)
        
        if is_blacklisted:
            reasons.append("Account is blacklisted")
        if is_suspended:
            reasons.append("Account is suspended")
        if claim_count >= 5:
            reasons.append(f"Very high claim count ({claim_count})")
        elif claim_count >= 4:
            reasons.append(f"High claim count ({claim_count})")
        if days_since < 7:
            reasons.append(f"Very recent claim ({days_since} days ago)")
        elif days_since < 15:
            reasons.append(f"Recent claim ({days_since} days ago)")
        
        hour = datetime.now().hour
        if 0 <= hour <= 5:
            reasons.append("Off-hours transaction (midnight-5am)")
        
        if fraud_prob >= 0.75 and not reasons:
            reasons.append("Unusual transaction pattern detected by AI")
        
        return {
            'fraud_probability': round(fraud_prob, 3),
            'is_suspicious': fraud_prob >= threshold,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'reasons': reasons,
            'model_used': 'ML (Calibrated Isolation Forest)' if self.is_fraud_model_loaded else 'Rule-Based'
        }
    
    def analyze_ledger_entry(self, ledger_data: Dict) -> Dict:
        """
        Analyze ledger entry for tampering
        """
        if not self.is_ledger_model_loaded:
            return self._rule_based_ledger_check(ledger_data)
        
        try:
            features = np.array([[
                ledger_data.get('hash_valid', 1),
                ledger_data.get('timestamp_gap', 50),
                ledger_data.get('checksum_valid', 1),
                ledger_data.get('field_count', 8)
            ]])
            
            prediction = self.ledger_model.predict(features)[0]
            probabilities = self.ledger_model.predict_proba(features)[0]
            
            tamper_prob = probabilities[1]
            is_tampered = prediction == 1
            
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
        """Rule-based ledger check"""
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


# Backward compatibility
class FraudDetector(ProductionFraudDetector):
    """Alias for backward compatibility"""
    pass


class EnhancedFraudDetector(ProductionFraudDetector):
    """Alias for backward compatibility"""
    pass