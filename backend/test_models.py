"""
Jan-Dhan Gateway - Production Model Testing
Test properly calibrated ML models
"""

import pickle
import numpy as np
import pandas as pd
import os

class ProductionModelTester:
    """Test production-calibrated models"""
    
    def __init__(self, models_dir: str = '../models'):
        self.models_dir = models_dir
        self.load_models()
    
    def load_models(self):
        """Load calibrated models"""
        print("📦 Loading production models...")
        
        try:
            # Load calibrated Isolation Forest
            with open(os.path.join(self.models_dir, 'fraud_model_calibrated.pkl'), 'rb') as f:
                fraud_data = pickle.load(f)
                self.fraud_model = fraud_data['model']
                self.fraud_scaler = fraud_data['scaler']
                self.min_score = fraud_data['min_score']
                self.max_score = fraud_data['max_score']
                self.features = fraud_data['features']
            
            print("✓ Calibrated Isolation Forest loaded")
            
            # Load supervised model
            with open(os.path.join(self.models_dir, 'fraud_model_supervised.pkl'), 'rb') as f:
                supervised_data = pickle.load(f)
                self.supervised_model = supervised_data['model']
                self.supervised_scaler = supervised_data['scaler']
            
            print("✓ Supervised Random Forest loaded")
            
            # Load ledger model
            with open(os.path.join(self.models_dir, 'ledger_tamper_model_calibrated.pkl'), 'rb') as f:
                self.ledger_model = pickle.load(f)
            
            print("✓ Ledger tamper model loaded")
            
        except FileNotFoundError as e:
            print(f"❌ Model not found: {e}")
            print("Run train_models_production.py first")
            exit(1)
    
    def predict_fraud_probability(self, transaction_data):
        """Predict fraud with proper calibration"""
        # Extract features
        features_array = [
            transaction_data.get('claim_count', 0),
            transaction_data.get('days_since_claim', 30),
            transaction_data.get('amount', 3000),
            transaction_data.get('hour', 12),
            transaction_data.get('day_of_week', 2),
            transaction_data.get('frequency_score', 0.01),
            transaction_data.get('is_suspended', 0),
            transaction_data.get('is_blacklisted', 0)
        ]
        
        X = np.array([features_array])
        X_scaled = self.fraud_scaler.transform(X)
        
        # Get anomaly score
        anomaly_score = self.fraud_model.score_samples(X_scaled)[0]
        
        # Calibrated probability calculation
        normalized_score = (anomaly_score - self.min_score) / (self.max_score - self.min_score)
        fraud_prob = 1 - normalized_score
        fraud_prob = max(0.0, min(1.0, fraud_prob))
        
        return fraud_prob
    
    def test_fraud_detection(self):
        """Test fraud detection with realistic cases"""
        print("\n" + "="*70)
        print("🧪 TESTING CALIBRATED FRAUD DETECTION")
        print("="*70)
        
        test_cases = [
            {
                'name': '✅ Normal Transaction - Low Claims',
                'data': {
                    'claim_count': 1,
                    'days_since_claim': 60,
                    'amount': 3000,
                    'hour': 14,
                    'day_of_week': 2,
                    'frequency_score': 1/60,
                    'is_suspended': 0,
                    'is_blacklisted': 0
                },
                'expected': 'LOW RISK'
            },
            {
                'name': '✅ Normal Transaction - Moderate Claims',
                'data': {
                    'claim_count': 2,
                    'days_since_claim': 90,
                    'amount': 5000,
                    'hour': 10,
                    'day_of_week': 1,
                    'frequency_score': 2/90,
                    'is_suspended': 0,
                    'is_blacklisted': 0
                },
                'expected': 'LOW RISK'
            },
            {
                'name': '⚠️  Suspicious - High Frequency',
                'data': {
                    'claim_count': 5,
                    'days_since_claim': 5,
                    'amount': 5000,
                    'hour': 10,
                    'day_of_week': 1,
                    'frequency_score': 1.0,
                    'is_suspended': 0,
                    'is_blacklisted': 0
                },
                'expected': 'HIGH RISK'
            },
            {
                'name': '🚨 Fraud - Off Hours + High Frequency',
                'data': {
                    'claim_count': 6,
                    'days_since_claim': 2,
                    'amount': 5000,
                    'hour': 3,
                    'day_of_week': 6,
                    'frequency_score': 3.0,
                    'is_suspended': 0,
                    'is_blacklisted': 0
                },
                'expected': 'CRITICAL'
            },
            {
                'name': '🚨 Fraud - Suspended Account',
                'data': {
                    'claim_count': 3,
                    'days_since_claim': 10,
                    'amount': 3000,
                    'hour': 14,
                    'day_of_week': 2,
                    'frequency_score': 0.3,
                    'is_suspended': 1,
                    'is_blacklisted': 0
                },
                'expected': 'HIGH RISK'
            }
        ]
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n{'='*70}")
            print(f"Test {i}: {test['name']}")
            print(f"{'='*70}")
            
            fraud_prob = self.predict_fraud_probability(test['data'])
            
            # Classify risk
            if fraud_prob >= 0.9:
                risk = "🔴 CRITICAL"
                recommendation = "REJECT"
            elif fraud_prob >= 0.75:
                risk = "🟠 HIGH"
                recommendation = "MANUAL_REVIEW"
            elif fraud_prob >= 0.5:
                risk = "🟡 MEDIUM"
                recommendation = "APPROVE with monitoring"
            else:
                risk = "🟢 LOW"
                recommendation = "APPROVE"
            
            print(f"Fraud Probability: {fraud_prob:.1%}")
            print(f"Risk Level: {risk}")
            print(f"Recommendation: {recommendation}")
            print(f"Expected: {test['expected']}")
            
            # Match check
            expected_lower = test['expected'].lower()
            risk_lower = risk.lower()
            match = "✅ MATCH" if expected_lower in risk_lower or risk_lower in expected_lower else "❌ MISMATCH"
            print(f"Result: {match}")
    
    def test_supervised_model(self):
        """Test supervised Random Forest model"""
        print("\n" + "="*70)
        print("🧪 TESTING SUPERVISED RANDOM FOREST")
        print("="*70)
        
        test_cases = [
            {
                'name': 'Normal - Low Risk',
                'data': [1, 60, 3000, 14, 2, 0.017, 0, 0],
                'expected': 'Normal'
            },
            {
                'name': 'Fraud - High Frequency',
                'data': [6, 5, 5000, 3, 6, 1.2, 0, 0],
                'expected': 'Fraud'
            }
        ]
        
        for test in test_cases:
            X = np.array([test['data']])
            X_scaled = self.supervised_scaler.transform(X)
            
            prediction = self.supervised_model.predict(X_scaled)[0]
            probability = self.supervised_model.predict_proba(X_scaled)[0]
            
            result = "Fraud" if prediction == 1 else "Normal"
            fraud_prob = probability[1]
            
            print(f"\n{test['name']}:")
            print(f"  Prediction: {result}")
            print(f"  Fraud Probability: {fraud_prob:.1%}")
            print(f"  Expected: {test['expected']}")
    
    def test_batch_performance(self):
        """Test on batch data"""
        print("\n" + "="*70)
        print("🧪 BATCH PERFORMANCE TEST")
        print("="*70)
        
        try:
            df = pd.read_csv(os.path.join(self.models_dir, 'realistic_transactions.csv'))
            
            print(f"\nTesting on {len(df)} transactions...")
            
            # Prepare features
            X = df[self.features].values
            X_scaled = self.fraud_scaler.transform(X)
            y_true = df['is_fraud'].values
            
            # Predict
            predictions = self.fraud_model.predict(X_scaled)
            y_pred = (predictions == -1).astype(int)
            
            # Metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            print(f"\n📊 Performance Metrics:")
            print(f"  Accuracy:  {accuracy:.1%}")
            print(f"  Precision: {precision:.1%}")
            print(f"  Recall:    {recall:.1%}")
            print(f"  F1-Score:  {f1:.1%}")
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            print(f"\n📈 Confusion Matrix:")
            print(f"  True Negatives:  {tn} (correctly identified normal)")
            print(f"  False Positives: {fp} (normal flagged as fraud)")
            print(f"  False Negatives: {fn} (fraud missed)")
            print(f"  True Positives:  {tp} (correctly identified fraud)")
            
        except FileNotFoundError:
            print("⚠️  Batch data not found")
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "="*70)
        print("🚀 JAN-DHAN GATEWAY - PRODUCTION MODEL TESTING")
        print("="*70)
        
        self.test_fraud_detection()
        self.test_supervised_model()
        self.test_batch_performance()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETED")
        print("="*70)


if __name__ == '__main__':
    tester = ProductionModelTester()
    tester.run_all_tests()