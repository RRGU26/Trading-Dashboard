#!/usr/bin/env python3
"""
Signal Database Integrator
For models that generate signals rather than price predictions
"""

import sqlite3
import json
from datetime import datetime, timedelta

def save_signal_prediction(model_name, symbol, signal_type, gmi_score=None,
                          signal_strength=None, current_price=None, target_return=None,
                          horizon_days=5, additional_data=None):
    """
    Save a signal-based prediction to the database

    Args:
        model_name: Name of the model (e.g., "Wishing Well QQQ Model")
        symbol: Trading symbol (e.g., "QQQ")
        signal_type: "BUY", "SELL", or "HOLD"
        gmi_score: GMI score (0-6) for Wishing Well model
        signal_strength: Confidence/strength of signal (0-100)
        current_price: Current price of the symbol
        target_return: Expected return percentage based on signal
        horizon_days: Number of days for the signal
        additional_data: Additional data as dictionary (will be JSON encoded)
    """

    try:
        conn = sqlite3.connect('reports_tracking.db')
        cursor = conn.cursor()

        # Ensure signal_predictions table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                symbol TEXT NOT NULL,
                prediction_date DATE NOT NULL,
                target_date DATE NOT NULL,
                signal_type TEXT NOT NULL,
                gmi_score INTEGER,
                signal_strength REAL,
                current_price REAL,
                target_return REAL,
                actual_price REAL,
                actual_return REAL,
                signal_correct INTEGER,
                horizon_days INTEGER DEFAULT 5,
                additional_data TEXT,
                created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        prediction_date = datetime.now().strftime('%Y-%m-%d')
        target_date = (datetime.now() + timedelta(days=horizon_days)).strftime('%Y-%m-%d')

        # Convert additional_data to JSON if provided
        additional_json = json.dumps(additional_data) if additional_data else None

        cursor.execute("""
            INSERT INTO signal_predictions
            (model, symbol, prediction_date, target_date, signal_type, gmi_score,
             signal_strength, current_price, target_return, horizon_days, additional_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_name, symbol, prediction_date, target_date, signal_type,
            gmi_score, signal_strength, current_price, target_return,
            horizon_days, additional_json
        ))

        conn.commit()
        conn.close()

        return True

    except Exception as e:
        print(f"Error saving signal prediction: {e}")
        return False

def update_signal_performance(model_name):
    """Update signal performance metrics for a model"""

    try:
        conn = sqlite3.connect('reports_tracking.db')
        cursor = conn.cursor()

        # Ensure signal_performance table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                total_signals INTEGER DEFAULT 0,
                correct_signals INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.0,
                avg_return REAL DEFAULT 0.0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Update performance metrics
        cursor.execute("""
            INSERT OR REPLACE INTO signal_performance
            (model, signal_type, total_signals, correct_signals, win_rate, avg_return, last_updated)
            SELECT
                model,
                signal_type,
                COUNT(*) as total,
                SUM(CASE WHEN signal_correct = 1 THEN 1 ELSE 0 END) as correct,
                CAST(SUM(CASE WHEN signal_correct = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) * 100 as win_rate,
                AVG(actual_return) as avg_return,
                datetime('now') as last_updated
            FROM signal_predictions
            WHERE model = ? AND actual_return IS NOT NULL
            GROUP BY model, signal_type
        """, (model_name,))

        conn.commit()
        conn.close()

        return True

    except Exception as e:
        print(f"Error updating signal performance: {e}")
        return False

def get_signal_performance(model_name):
    """Get signal performance metrics for a model"""

    try:
        conn = sqlite3.connect('reports_tracking.db')
        cursor = conn.cursor()

        cursor.execute("""
            SELECT signal_type, total_signals, correct_signals, win_rate, avg_return
            FROM signal_performance
            WHERE model = ?
            ORDER BY total_signals DESC
        """, (model_name,))

        results = cursor.fetchall()
        conn.close()

        return results

    except Exception as e:
        print(f"Error getting signal performance: {e}")
        return []