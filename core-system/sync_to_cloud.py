#!/usr/bin/env python3
"""
Sync predictions data to cloud dashboard via GitHub
Runs after each model execution
"""

import sqlite3
import json
import os
import subprocess
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def export_predictions_to_json():
    """Export latest predictions to JSON for cloud dashboard"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "models_dashboard.db")
    
    if not os.path.exists(db_path):
        logger.error("Database not found")
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get latest predictions (today's data)
        today = datetime.now().strftime('%Y-%m-%d')
        
        query = """
        SELECT 
            model,
            symbol,
            current_price,
            predicted_price,
            suggested_action,
            confidence,
            expected_return,
            prediction_date,
            target_date
        FROM model_predictions
        WHERE prediction_date = ?
        ORDER BY model
        """
        
        cursor.execute(query, (today,))
        rows = cursor.fetchall()
        
        # Convert to list of dictionaries with data cleaning
        predictions = []
        columns = [description[0] for description in cursor.description]
        
        for row in rows:
            prediction = dict(zip(columns, row))
            
            # Clean up binary/bytes data that shouldn't be there
            for key, value in prediction.items():
                if isinstance(value, (bytes, memoryview)):
                    # Try to convert bytes to float if it's a numeric field
                    if key in ['predicted_price', 'expected_return', 'current_price']:
                        try:
                            # Unpack as float (assuming little-endian 4-byte float)
                            import struct
                            if len(value) == 4:
                                prediction[key] = struct.unpack('<f', value)[0]
                            else:
                                prediction[key] = None
                        except:
                            prediction[key] = None
                    else:
                        # For other fields, try to decode as string
                        try:
                            prediction[key] = value.decode('utf-8')
                        except:
                            prediction[key] = str(value)
                            
            predictions.append(prediction)
        
        # Add metadata
        export_data = {
            'last_updated': datetime.now().isoformat(),
            'export_date': today,
            'total_predictions': len(predictions),
            'predictions': predictions,
            'system_status': 'operational',
            'next_update': '15:40 ET daily'
        }
        
        # Save to JSON file
        json_path = os.path.join(script_dir, '..', 'data', 'latest_predictions.json')
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        conn.close()
        
        logger.info(f"Exported {len(predictions)} predictions to {json_path}")
        return json_path
        
    except Exception as e:
        logger.error(f"Error exporting predictions: {e}")
        return None

def sync_to_github():
    """Push latest data to GitHub for cloud dashboard"""
    try:
        # Change to repo directory
        repo_dir = os.path.join(os.path.dirname(__file__), '..')
        os.chdir(repo_dir)
        
        # Git commands
        commands = [
            ['git', 'add', 'data/latest_predictions.json'],
            ['git', 'commit', '-m', f'Auto-update predictions - {datetime.now().strftime("%Y-%m-%d %H:%M")}'],
            ['git', 'push', 'origin', 'master']
        ]
        
        for cmd in commands:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"Git command failed: {' '.join(cmd)} - {result.stderr}")
                # Don't fail completely - commit might be empty
        
        logger.info("Successfully synced data to GitHub")
        return True
        
    except Exception as e:
        logger.error(f"Error syncing to GitHub: {e}")
        return False

def create_summary_json():
    """Create a summary JSON with key metrics including performance tracking"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "models_dashboard.db")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Get signal counts
        cursor.execute("""
            SELECT suggested_action, COUNT(*) 
            FROM model_predictions 
            WHERE prediction_date = ?
            GROUP BY suggested_action
        """, (today,))
        
        signal_counts = dict(cursor.fetchall())
        
        # Get average confidence
        cursor.execute("""
            SELECT AVG(confidence) 
            FROM model_predictions 
            WHERE prediction_date = ?
        """, (today,))
        
        avg_confidence = cursor.fetchone()[0] or 0
        
        # Get recent accuracy data (last 30 days)
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        cursor.execute("""
            SELECT 
                model,
                COUNT(*) as total_predictions,
                AVG(CASE WHEN direction_correct = 1 THEN 1.0 ELSE 0.0 END) * 100 as accuracy,
                AVG(ABS(error_pct)) as avg_error
            FROM model_predictions 
            WHERE prediction_date >= ? AND actual_price IS NOT NULL
            GROUP BY model
        """, (thirty_days_ago,))
        
        performance_data = []
        for row in cursor.fetchall():
            performance_data.append({
                'model': row[0],
                'total_predictions': row[1],
                'accuracy': round(row[2] or 0, 1),
                'avg_error': round(row[3] or 0, 2)
            })
        
        # Create summary
        summary = {
            'date': today,
            'last_updated': datetime.now().isoformat(),
            'signal_counts': signal_counts,
            'average_confidence': round(avg_confidence, 1),
            'total_models': len(signal_counts) if signal_counts else 0,
            'status': 'active' if signal_counts else 'pending',
            'performance_data': performance_data,
            'dashboard_url': 'https://rrgu26-trading-dashboard-cloud-dashboard-with-sync-crrtqv.streamlit.app'
        }
        
        # Save summary
        summary_path = os.path.join(script_dir, '..', 'data', 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        conn.close()
        return summary_path
        
    except Exception as e:
        logger.error(f"Error creating summary: {e}")
        return None

def main():
    """Main sync function - call after model runs"""
    logger.info("Starting cloud sync process...")
    
    # Export latest predictions
    json_path = export_predictions_to_json()
    if not json_path:
        logger.error("Failed to export predictions")
        return False
    
    # Create summary
    summary_path = create_summary_json()
    if summary_path:
        logger.info("Summary created successfully")
    
    # Sync to GitHub (triggers Streamlit Cloud auto-deploy)
    success = sync_to_github()
    
    if success:
        logger.info("Cloud sync completed successfully")
        print("SUCCESS: Dashboard data synced to cloud")
    else:
        logger.error("Cloud sync failed")
        print("FAILED: Failed to sync to cloud")
    
    return success

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()