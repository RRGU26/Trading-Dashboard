#!/usr/bin/env python3
"""
Fix model_predictions table schema by adding missing columns
"""
import sqlite3
import os

def fix_database_schema():
    """Add missing columns to model_predictions table"""
    
    db_path = os.path.join(os.path.dirname(__file__), "models_dashboard.db")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("Checking current schema...")
        cursor.execute('PRAGMA table_info(model_predictions)')
        columns = cursor.fetchall()
        existing_columns = [col[1] for col in columns]
        
        print(f"Existing columns: {existing_columns}")
        
        # Add missing columns
        missing_columns = []
        
        if 'expected_return' not in existing_columns:
            cursor.execute('ALTER TABLE model_predictions ADD COLUMN expected_return REAL')
            missing_columns.append('expected_return')
            print("Added expected_return column")
        
        if 'actual_return' not in existing_columns:
            cursor.execute('ALTER TABLE model_predictions ADD COLUMN actual_return REAL')
            missing_columns.append('actual_return')
            print("Added actual_return column")
            
        if 'return_error' not in existing_columns:
            cursor.execute('ALTER TABLE model_predictions ADD COLUMN return_error REAL')
            missing_columns.append('return_error')
            print("Added return_error column")
        
        conn.commit()
        
        if missing_columns:
            print(f"Successfully added columns: {missing_columns}")
        else:
            print("All required columns already exist")
        
        # Verify the fix
        cursor.execute('PRAGMA table_info(model_predictions)')
        columns = cursor.fetchall()
        print(f"Updated schema:")
        for col in columns:
            print(f"  {col[1]} ({col[2]})")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error fixing database schema: {e}")
        return False

if __name__ == '__main__':
    fix_database_schema()