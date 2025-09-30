#!/usr/bin/env python3
"""
COMPREHENSIVE DATA SALVAGE TOOL
===============================
Salvages all prediction data from backup databases into the master database
Ensures maximum data completeness for the trading system
"""

import sqlite3
import os
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataSalvageEngine:
    """Comprehensive data salvage and consolidation engine"""
    
    def __init__(self, master_db_path: str):
        self.master_db_path = master_db_path
        self.desktop_path = Path(r"C:\Users\rrose\OneDrive\Desktop")
        self.backup_sources = []
        self.salvaged_records = 0
        
    def find_all_backup_databases(self):
        """Find all backup database files"""
        logger.info("🔍 Scanning for backup databases...")
        
        # Search patterns for database files
        search_patterns = [
            "models_dashboard_backup_*.db",
            "reports_tracking.db", 
            "qqq_master_model.db",
            "reports_tracking.db",
            "**/reports_tracking.db"  # Recursive search
        ]
        
        backup_files = []
        
        for pattern in search_patterns:
            found_files = list(self.desktop_path.rglob(pattern))
            backup_files.extend(found_files)
        
        # Remove duplicates and sort by modification time (newest first)
        unique_files = list(set(backup_files))
        unique_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Filter out the master database itself
        master_path = Path(self.master_db_path)
        unique_files = [f for f in unique_files if f != master_path]
        
        logger.info(f"📁 Found {len(unique_files)} backup database files:")
        for i, file in enumerate(unique_files[:10]):  # Show first 10
            size_mb = file.stat().st_size / (1024*1024)
            mod_time = datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            logger.info(f"  {i+1}. {file.name} ({size_mb:.1f}MB, {mod_time})")
        
        if len(unique_files) > 10:
            logger.info(f"  ... and {len(unique_files)-10} more backup files")
        
        self.backup_sources = unique_files
        return unique_files
    
    def check_database_integrity(self, db_path: Path) -> bool:
        """Check if database is accessible and not corrupted"""
        try:
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                table_count = cursor.fetchone()[0]
                
                if table_count == 0:
                    logger.warning(f"⚠️ {db_path.name}: No tables found")
                    return False
                
                # Try to access model_predictions table
                try:
                    cursor.execute("SELECT COUNT(*) FROM model_predictions")
                    record_count = cursor.fetchone()[0]
                    logger.info(f"✅ {db_path.name}: {record_count} predictions, {table_count} tables")
                    return True
                except sqlite3.Error as e:
                    logger.warning(f"⚠️ {db_path.name}: Table access error - {e}")
                    return False
                    
        except sqlite3.Error as e:
            logger.error(f"❌ {db_path.name}: Database error - {e}")
            return False
        except Exception as e:
            logger.error(f"❌ {db_path.name}: Unexpected error - {e}")
            return False
    
    def get_database_schema(self, db_path: Path) -> dict:
        """Get the schema information for a database"""
        try:
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.cursor()
                
                # Get table schemas
                cursor.execute("""
                    SELECT name, sql FROM sqlite_master 
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """)
                
                tables = {}
                for table_name, create_sql in cursor.fetchall():
                    # Get column info
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()
                    
                    tables[table_name] = {
                        'create_sql': create_sql,
                        'columns': columns
                    }
                
                return tables
                
        except Exception as e:
            logger.error(f"Error getting schema for {db_path.name}: {e}")
            return {}
    
    def salvage_predictions_from_database(self, source_db_path: Path) -> int:
        """Salvage prediction data from a single backup database"""
        logger.info(f"💾 Salvaging data from: {source_db_path.name}")
        
        try:
            # Check source database integrity
            if not self.check_database_integrity(source_db_path):
                logger.warning(f"⚠️ Skipping corrupted database: {source_db_path.name}")
                return 0
            
            # Connect to both databases
            source_conn = sqlite3.connect(str(source_db_path))
            master_conn = sqlite3.connect(self.master_db_path)
            
            # Get existing prediction IDs from master to avoid duplicates
            master_cursor = master_conn.cursor()
            master_cursor.execute("SELECT model, symbol, prediction_date, target_date FROM model_predictions")
            existing_predictions = set(master_cursor.fetchall())
            
            # Fetch all predictions from source
            source_cursor = source_conn.cursor()
            source_cursor.execute("""
                SELECT * FROM model_predictions 
                ORDER BY prediction_date, created_timestamp
            """)
            
            source_predictions = source_cursor.fetchall()
            
            if not source_predictions:
                logger.info(f"📭 No predictions found in {source_db_path.name}")
                source_conn.close()
                master_conn.close()
                return 0
            
            # Get column names
            source_cursor.execute("PRAGMA table_info(model_predictions)")
            columns = [col[1] for col in source_cursor.fetchall()]
            
            # Insert unique predictions
            new_predictions = 0
            insert_sql = f"""
                INSERT OR IGNORE INTO model_predictions 
                ({','.join(columns)}) VALUES ({','.join(['?' for _ in columns])})
            """
            
            for prediction in source_predictions:
                # Check for duplicates based on key fields
                key = (prediction[columns.index('model')] if 'model' in columns else '',
                       prediction[columns.index('symbol')] if 'symbol' in columns else '',
                       prediction[columns.index('prediction_date')] if 'prediction_date' in columns else '',
                       prediction[columns.index('target_date')] if 'target_date' in columns else '')
                
                if key not in existing_predictions:
                    try:
                        master_cursor.execute(insert_sql, prediction)
                        new_predictions += 1
                        existing_predictions.add(key)
                    except sqlite3.Error as e:
                        logger.warning(f"⚠️ Error inserting prediction: {e}")
            
            master_conn.commit()
            source_conn.close()
            master_conn.close()
            
            logger.info(f"✅ Salvaged {new_predictions} new predictions from {source_db_path.name}")
            return new_predictions
            
        except Exception as e:
            logger.error(f"❌ Error salvaging from {source_db_path.name}: {e}")
            return 0
    
    def consolidate_price_history(self):
        """Consolidate price history from all sources"""
        logger.info("💰 Consolidating price history data...")
        
        total_prices_added = 0
        
        for source_db_path in self.backup_sources[:5]:  # Process top 5 most recent
            try:
                source_conn = sqlite3.connect(str(source_db_path))
                source_cursor = source_conn.cursor()
                
                # Check if price_history table exists
                source_cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='price_history'
                """)
                
                if not source_cursor.fetchone():
                    source_conn.close()
                    continue
                
                # Get price history data
                source_cursor.execute("SELECT * FROM price_history ORDER BY date")
                price_data = source_cursor.fetchall()
                
                if price_data:
                    # Insert into master database
                    master_conn = sqlite3.connect(self.master_db_path)
                    master_cursor = master_conn.cursor()
                    
                    source_cursor.execute("PRAGMA table_info(price_history)")
                    columns = [col[1] for col in source_cursor.fetchall()]
                    
                    insert_sql = f"""
                        INSERT OR IGNORE INTO price_history 
                        ({','.join(columns)}) VALUES ({','.join(['?' for _ in columns])})
                    """
                    
                    prices_added = 0
                    for price_row in price_data:
                        try:
                            master_cursor.execute(insert_sql, price_row)
                            prices_added += 1
                        except sqlite3.Error:
                            pass  # Ignore duplicates
                    
                    master_conn.commit()
                    master_conn.close()
                    
                    if prices_added > 0:
                        logger.info(f"💰 Added {prices_added} price records from {source_db_path.name}")
                        total_prices_added += prices_added
                
                source_conn.close()
                
            except Exception as e:
                logger.warning(f"⚠️ Error processing price history from {source_db_path.name}: {e}")
        
        logger.info(f"💰 Total price records consolidated: {total_prices_added}")
        return total_prices_added
    
    def run_comprehensive_salvage(self) -> dict:
        """Run complete data salvage operation"""
        logger.info("🚀 STARTING COMPREHENSIVE DATA SALVAGE")
        logger.info("=" * 60)
        
        results = {
            'databases_found': 0,
            'databases_processed': 0,
            'predictions_salvaged': 0,
            'price_records_added': 0,
            'errors': []
        }
        
        # Find all backup databases
        backup_files = self.find_all_backup_databases()
        results['databases_found'] = len(backup_files)
        
        if not backup_files:
            logger.warning("⚠️ No backup databases found")
            return results
        
        # Process each backup database
        for i, backup_db in enumerate(backup_files, 1):
            logger.info(f"📊 Processing database {i}/{len(backup_files)}: {backup_db.name}")
            
            try:
                predictions_added = self.salvage_predictions_from_database(backup_db)
                results['predictions_salvaged'] += predictions_added
                results['databases_processed'] += 1
                
            except Exception as e:
                error_msg = f"Error processing {backup_db.name}: {e}"
                logger.error(f"❌ {error_msg}")
                results['errors'].append(error_msg)
        
        # Consolidate price history
        price_records = self.consolidate_price_history()
        results['price_records_added'] = price_records
        
        # Final summary
        logger.info("=" * 60)
        logger.info("🎉 DATA SALVAGE COMPLETE!")
        logger.info(f"📊 Databases Found: {results['databases_found']}")
        logger.info(f"✅ Databases Processed: {results['databases_processed']}")
        logger.info(f"💾 Predictions Salvaged: {results['predictions_salvaged']}")
        logger.info(f"💰 Price Records Added: {results['price_records_added']}")
        logger.info(f"❌ Errors: {len(results['errors'])}")
        logger.info("=" * 60)
        
        return results

def main():
    """Main execution function"""
    # Set paths
    desktop_path = Path(r"C:\Users\rrose\OneDrive\Desktop")
    master_db_path = str(desktop_path / "reports_tracking.db")
    
    # Create backup of current master database
    if os.path.exists(master_db_path):
        backup_path = str(desktop_path / f"models_dashboard_pre_salvage_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
        shutil.copy2(master_db_path, backup_path)
        logger.info(f"💾 Created backup of master database: {Path(backup_path).name}")
    
    # Run salvage operation
    salvage_engine = DataSalvageEngine(master_db_path)
    results = salvage_engine.run_comprehensive_salvage()
    
    # Copy salvaged database to GitHub structure
    github_db_path = r"C:\Users\rrose\trading-models-system\databases\reports_tracking.db"
    if os.path.exists(master_db_path):
        shutil.copy2(master_db_path, github_db_path)
        logger.info(f"📁 Copied salvaged database to GitHub: {Path(github_db_path).name}")
    
    return results

if __name__ == "__main__":
    main()