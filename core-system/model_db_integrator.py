#!/usr/bin/env python3
"""
MODEL DATABASE INTEGRATOR
========================
Universal database integration module for trading models
Adds database saving capability to any model
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import traceback

class ModelDatabaseIntegrator:
    """Universal database integrator for trading models"""
    
    def __init__(self, model_name: str, symbol: str):
        self.model_name = model_name
        self.symbol = symbol
        self.db_path = self.find_database()
        
    def find_database(self) -> Optional[str]:
        """Find the models dashboard database"""
        possible_paths = [
            os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "reports_tracking.db"),
            os.path.join(os.path.expanduser("~"), "Desktop", "reports_tracking.db"),
            "reports_tracking.db",
            os.path.join(".", "reports_tracking.db")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"[OK] Found database: {path}")
                return path
        
        print("[ERROR] Database not found. Predictions will not be saved.")
        return None
    
    def save_prediction(self, 
                       current_price: float,
                       predicted_price: float, 
                       confidence: float,
                       horizon_days: int = 1,
                       suggested_action: str = "HOLD") -> bool:
        """Save prediction to model_predictions table"""
        if not self.db_path:
            return False
            
        try:
            prediction_date = datetime.now().strftime('%Y-%m-%d')
            target_date = (datetime.now() + timedelta(days=horizon_days)).strftime('%Y-%m-%d')
            
            # Calculate expected return
            predicted_return = ((predicted_price - current_price) / current_price) * 100
            
            # Determine action if not provided
            if suggested_action == "HOLD":
                if predicted_return > 2.0:
                    suggested_action = "BUY"
                elif predicted_return < -2.0:
                    suggested_action = "SELL"
                else:
                    suggested_action = "HOLD"
            
            # Use timeout and WAL mode to handle concurrent access
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute('PRAGMA journal_mode=WAL')  # Enable Write-Ahead Logging
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO model_predictions (
                    model, symbol, prediction_date, target_date, horizon,
                    current_price, predicted_price, actual_price, confidence,
                    suggested_action, error_pct, direction_correct,
                    expected_return, actual_return, return_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.model_name, self.symbol, prediction_date, target_date, horizon_days,
                float(current_price), float(predicted_price), None, float(confidence),
                suggested_action, None, None, float(predicted_return), None, None
            ))
            
            conn.commit()
            conn.close()
            
            print(f"[OK] Prediction saved to database")
            print(f"   Date: {prediction_date} -> {target_date} ({horizon_days}d)")
            print(f"   Price: ${current_price:.4f} -> ${predicted_price:.4f} ({predicted_return:+.2f}%)")
            print(f"   Action: {suggested_action} (Confidence: {confidence:.1f}%)")
            
            return True
            
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                print(f"[RETRY] Database locked, retrying in 5 seconds...")
                import time
                time.sleep(5)
                # Try one more time
                try:
                    conn = sqlite3.connect(self.db_path, timeout=60.0)
                    conn.execute('PRAGMA journal_mode=WAL')
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        INSERT INTO model_predictions (
                            model, symbol, prediction_date, target_date, horizon,
                            current_price, predicted_price, actual_price, confidence,
                            suggested_action, error_pct, direction_correct,
                            expected_return, actual_return, return_error
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        self.model_name, self.symbol, prediction_date, target_date, horizon_days,
                        float(current_price), float(predicted_price), None, float(confidence),
                        suggested_action, None, None, float(predicted_return), None, None
                    ))
                    
                    conn.commit()
                    conn.close()
                    print(f"[OK] Prediction saved to database (retry successful)")
                    return True
                except Exception as retry_e:
                    print(f"[ERROR] Retry failed: {retry_e}")
                    return False
            else:
                print(f"[ERROR] Database error: {e}")
                return False
        except Exception as e:
            print(f"[ERROR] Error saving prediction: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            return False
    
    def save_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Save model metrics to model_metrics table"""
        if not self.db_path or not metrics:
            return False
            
        try:
            # Use timeout and WAL mode to handle concurrent access
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute('PRAGMA journal_mode=WAL')  # Enable Write-Ahead Logging
            cursor = conn.cursor()
            
            date_str = datetime.now().strftime('%Y-%m-%d')
            
            for metric_name, metric_value in metrics.items():
                cursor.execute("""
                    INSERT INTO model_metrics (model, date, metric_type, metric_value)
                    VALUES (?, ?, ?, ?)
                """, (self.model_name, date_str, metric_name, float(metric_value)))
            
            conn.commit()
            conn.close()
            
            print(f"[OK] Model metrics saved to database")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error saving metrics: {e}")
            return False
    
    def update_actual_prices(self) -> int:
        """Update actual prices for resolved predictions"""
        if not self.db_path:
            return 0
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find predictions that need actual price updates
            cursor.execute("""
                SELECT id, current_price, predicted_price, target_date 
                FROM model_predictions 
                WHERE model = ? AND actual_price IS NULL 
                AND target_date <= date('now')
            """, (self.model_name,))
            
            pending_predictions = cursor.fetchall()
            updated_count = 0
            
            # For now, we'll update with a placeholder
            # In a real implementation, you'd fetch actual prices from an API
            for pred_id, current_price, predicted_price, target_date in pending_predictions:
                # Placeholder: assume some random price movement for demonstration
                # In practice, fetch real price data here
                pass
            
            conn.close()
            return updated_count
            
        except Exception as e:
            print(f"[ERROR] Error updating actual prices: {e}")
            return 0

def quick_save_prediction(model_name: str, symbol: str, current_price: float, 
                         predicted_price: float, confidence: float = 80.0,
                         horizon_days: int = 1, suggested_action: str = "HOLD") -> bool:
    """Quick function to save a prediction from any model"""
    integrator = ModelDatabaseIntegrator(model_name, symbol)
    return integrator.save_prediction(
        current_price=current_price,
        predicted_price=predicted_price,
        confidence=confidence,
        horizon_days=horizon_days,
        suggested_action=suggested_action
    )

def quick_save_metrics(model_name: str, symbol: str, metrics: Dict[str, Any]) -> bool:
    """Quick function to save metrics from any model"""
    integrator = ModelDatabaseIntegrator(model_name, symbol)
    return integrator.save_metrics(metrics)

# Example usage for models:
if __name__ == "__main__":
    # Test the integrator
    print("Testing Model Database Integrator...")
    
    # Example prediction save
    success = quick_save_prediction(
        model_name="Test Model",
        symbol="BTC-USD", 
        current_price=45000.0,
        predicted_price=46000.0,
        confidence=85.0,
        horizon_days=1,
        suggested_action="BUY"
    )
    
    if success:
        print("[OK] Test prediction saved successfully!")
    else:
        print("[ERROR] Test prediction failed to save.")
    
    # Example metrics save
    test_metrics = {
        "r2_score": 0.75,
        "mae": 0.05,
        "hit_rate": 0.68
    }
    
    success = quick_save_metrics("Test Model", "BTC-USD", test_metrics)
    
    if success:
        print("[OK] Test metrics saved successfully!")
    else:
        print("[ERROR] Test metrics failed to save.")