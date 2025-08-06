import sqlite3
import os
import datetime
import glob
import logging
import time
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingReportsDatabase:
    """Database manager for trading reports and email tracking"""
    
    def __init__(self, db_path="reports_tracking.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table for tracking report files
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS report_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                model_type TEXT,
                generated_date DATE,
                file_size INTEGER,
                created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(filepath)
            )
        ''')
        
        # Table for tracking email sends
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS email_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                send_date DATE,
                email_subject TEXT,
                recipients TEXT,
                report_files_count INTEGER,
                status TEXT,
                error_message TEXT,
                sent_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table for report metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS report_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_date DATE,
                model_name TEXT,
                symbol TEXT,
                current_price REAL,
                predicted_price REAL,
                signal TEXT,
                confidence REAL,
                created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table for predictions (needed by other modules)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                symbol TEXT NOT NULL,
                prediction_date DATE NOT NULL,
                target_date DATE NOT NULL,
                current_price REAL,
                predicted_price REAL,
                expected_return REAL,
                actual_price REAL,
                actual_return REAL,
                direction_correct INTEGER,
                horizon_days INTEGER,
                created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table for price history (needed by other modules)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def scan_report_files(self, base_paths):
        """Scan for report files and add them to database"""
        if isinstance(base_paths, str):
            base_paths = [base_paths]
        
        report_patterns = [
            "*Long*Bull*Report*.txt",
            "*Trading*Signal*.txt", 
            "*Algorand*Report*.txt",
            "*algorand*report*.txt",
            "*Bitcoin*Prediction*Report*.txt",
            "*WishingWealth*.txt",
            "*NVIDIA*Report*.txt"
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        total_files = 0
        new_files = 0
        
        for base_path in base_paths:
            if not os.path.exists(base_path):
                continue
                
            for pattern in report_patterns:
                files = glob.glob(os.path.join(base_path, pattern))
                
                for filepath in files:
                    total_files += 1
                    
                    # Check if file already exists in database
                    cursor.execute(
                        "SELECT id FROM report_files WHERE filepath = ?", 
                        (filepath,)
                    )
                    
                    if cursor.fetchone() is None:
                        # Add new file to database
                        filename = os.path.basename(filepath)
                        file_size = os.path.getsize(filepath)
                        file_date = datetime.datetime.fromtimestamp(
                            os.path.getctime(filepath)
                        ).date()
                        
                        # Determine model type from filename
                        model_type = self._get_model_type(filename)
                        
                        cursor.execute('''
                            INSERT OR IGNORE INTO report_files 
                            (filename, filepath, model_type, generated_date, file_size)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (filename, filepath, model_type, file_date, file_size))
                        
                        new_files += 1
        
        conn.commit()
        conn.close()
        
        logger.info(f"Scanned {total_files} files, added {new_files} new files to database")
        return total_files, new_files
    
    def _get_model_type(self, filename):
        """Determine model type from filename"""
        filename_lower = filename.lower()
        
        if 'algorand' in filename_lower:
            return 'Algorand'
        elif 'bitcoin' in filename_lower:
            return 'Bitcoin'
        elif 'nvidia' in filename_lower:
            return 'NVIDIA'
        elif 'long' in filename_lower and 'bull' in filename_lower:
            return 'QQQ Long Bull'
        elif 'trading' in filename_lower and 'signal' in filename_lower:
            return 'QQQ Trading Signal'
        elif 'wishing' in filename_lower:
            return 'Wishing Well QQQ'
        else:
            return 'Unknown'
    
    def get_todays_reports(self, report_date=None):
        """Get all reports generated today"""
        if report_date is None:
            report_date = datetime.date.today()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT filepath, filename, model_type, file_size
            FROM report_files 
            WHERE generated_date = ?
            ORDER BY model_type, filename
        ''', (report_date,))
        
        reports = cursor.fetchall()
        conn.close()
        
        return reports
    
    def log_email_attempt(self, subject, recipients, report_count, status, error_msg=None):
        """Log an email sending attempt"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO email_history 
            (send_date, email_subject, recipients, report_files_count, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.date.today(),
            subject,
            str(recipients),
            report_count,
            status,
            error_msg
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Logged email attempt: {status}")
    
    def get_latest_reports_by_model(self):
        """Get the most recent report for each model type"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT rf.filepath, rf.filename, rf.model_type, rf.generated_date
            FROM report_files rf
            INNER JOIN (
                SELECT model_type, MAX(generated_date) as max_date
                FROM report_files 
                GROUP BY model_type
            ) latest ON rf.model_type = latest.model_type 
                    AND rf.generated_date = latest.max_date
            ORDER BY rf.model_type
        ''')
        
        reports = cursor.fetchall()
        conn.close()
        
        return reports
    
    def cleanup_old_records(self, days_to_keep=30):
        """Clean up old database records"""
        cutoff_date = datetime.date.today() - datetime.timedelta(days=days_to_keep)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clean up old email history
        cursor.execute(
            "DELETE FROM email_history WHERE send_date < ?", 
            (cutoff_date,)
        )
        
        # Clean up old report files (but keep the files themselves)
        cursor.execute(
            "DELETE FROM report_files WHERE generated_date < ?", 
            (cutoff_date,)
        )
        
        conn.commit()
        deleted_count = cursor.rowcount
        conn.close()
        
        logger.info(f"Cleaned up {deleted_count} old records")
        return deleted_count
    
    # Methods needed by other modules
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def backup_database(self):
        """Backup database"""
        try:
            backup_path = f"{self.db_path}.backup_{datetime.datetime.now().strftime('%Y%m%d')}"
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False
    
    def get_latest_price(self, symbol):
        """Get latest price for a symbol"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT close FROM price_history 
            WHERE symbol = ? 
            ORDER BY date DESC LIMIT 1
        ''', (symbol,))
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def get_historical_price(self, symbol, target_date):
        """Get historical price for a symbol on a specific date"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT close FROM price_history 
            WHERE symbol = ? AND date = ?
        ''', (symbol, target_date))
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def get_pending_predictions(self):
        """Get predictions that need actual price updates"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.date.today()
        cursor.execute('''
            SELECT id, symbol, target_date FROM model_predictions 
            WHERE target_date < ? AND actual_price IS NULL
        ''', (today,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [{'id': row[0], 'symbol': row[1], 'target_date': row[2]} for row in results]
    
    def update_prediction_actual(self, prediction_id, actual_price):
        """Update prediction with actual price"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE model_predictions 
                SET actual_price = ? 
                WHERE id = ?
            ''', (actual_price, prediction_id))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error updating prediction {prediction_id}: {e}")
            return False
    
    def store_prediction(self, model_name, symbol, current_price, predicted_price, expected_return, horizon_days=3):
        """Store a new prediction"""
        try:
            today = datetime.date.today()
            target_date = today + datetime.timedelta(days=horizon_days)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO model_predictions 
                (model, symbol, prediction_date, target_date, current_price, predicted_price, expected_return, horizon_days)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (model_name, symbol, today, target_date, current_price, predicted_price, expected_return, horizon_days))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error storing prediction for {model_name}: {e}")
            return False
    
    def get_model_performance(self):
        """Get model performance statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT model, symbol, 
                   COUNT(*) as total_predictions,
                   SUM(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) as correct_predictions
            FROM model_predictions 
            WHERE actual_price IS NOT NULL
            GROUP BY model, symbol
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        return [{'model': row[0], 'symbol': row[1], 'total_predictions': row[2], 'correct_predictions': row[3]} for row in results]


class DatabaseManager:
    """Global database manager for easy access"""
    
    def __init__(self):
        self.db = TradingReportsDatabase()
        self._status_cache = None
        self._last_status_check = None
    
    def initialize(self):
        """Initialize the database"""
        return self.db.init_database()
    
    def scan_reports(self, paths=None):
        """Scan for reports"""
        if paths is None:
            paths = [
                r"C:\Users\rrose\OneDrive\Desktop",
                r"C:\Users\rrose\Desktop"
            ]
        return self.db.scan_report_files(paths)
    
    def get_todays_reports(self):
        """Get today's reports"""
        return self.db.get_todays_reports()
    
    def log_email(self, subject, recipients, count, status, error=None):
        """Log email attempt"""
        return self.db.log_email_attempt(subject, recipients, count, status, error)
    
    def get_database_status(self):
        """Get database status with caching"""
        import time
        from datetime import datetime
        
        # Cache status for 60 seconds
        now = time.time()
        if (self._status_cache is None or 
            self._last_status_check is None or 
            now - self._last_status_check > 60):
            
            try:
                # Check if database file exists and is accessible
                if not os.path.exists(self.db.db_path):
                    status = "Database file not found"
                    updated = False
                else:
                    conn = sqlite3.connect(self.db.db_path)
                    cursor = conn.cursor()
                    
                    # Check for recent activity
                    today = datetime.now().date()
                    cursor.execute(
                        "SELECT COUNT(*) FROM report_files WHERE generated_date = ?",
                        (today,)
                    )
                    todays_files = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(*) FROM report_files")
                    total_files = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(*) FROM email_history")
                    total_emails = cursor.fetchone()[0]
                    
                    conn.close()
                    
                    status = f"Active - {total_files} files, {total_emails} emails, {todays_files} today"
                    updated = todays_files > 0
                
                self._status_cache = (updated, status)
                self._last_status_check = now
                
            except Exception as e:
                status = f"Database error: {str(e)[:50]}"
                self._status_cache = (False, status)
                self._last_status_check = now
        
        return self._status_cache
    
    # Delegate other methods to the database instance
    def get_connection(self):
        return self.db.get_connection()
    
    def backup_database(self):
        return self.db.backup_database()
    
    def get_latest_price(self, symbol):
        return self.db.get_latest_price(symbol)
    
    def get_historical_price(self, symbol, target_date):
        return self.db.get_historical_price(symbol, target_date)
    
    def get_pending_predictions(self):
        return self.db.get_pending_predictions()
    
    def update_prediction_actual(self, prediction_id, actual_price):
        return self.db.update_prediction_actual(prediction_id, actual_price)
    
    def store_prediction(self, model_name, symbol, current_price, predicted_price, expected_return, horizon_days=3):
        return self.db.store_prediction(model_name, symbol, current_price, predicted_price, expected_return, horizon_days)
    
    def get_model_performance(self):
        return self.db.get_model_performance()


# Create global database manager instance
db_manager = DatabaseManager()

# Convenience functions for easy import
def get_database(db_path="reports_tracking.db"):
    """Get a database instance"""
    return TradingReportsDatabase(db_path)

def scan_reports(base_paths):
    """Quick function to scan reports"""
    db = get_database()
    return db.scan_report_files(base_paths)

def get_todays_reports():
    """Quick function to get today's reports"""
    db = get_database()
    return db.get_todays_reports()

def log_email(subject, recipients, count, status, error=None):
    """Quick function to log email attempts"""
    db = get_database()
    return db.log_email_attempt(subject, recipients, count, status, error)

# Example usage and testing
if __name__ == "__main__":
    print("Testing TradingReportsDatabase...")
    
    # Test database manager
    print("Testing database manager...")
    db_manager.initialize()
    
    # Test scanning (adjust paths as needed)
    test_paths = [
        r"C:\Users\rrose\OneDrive\Desktop",
        r"C:\Users\rrose\Desktop"
    ]
    
    total, new = db_manager.scan_reports(test_paths)
    print(f"Found {total} files, {new} new")
    
    # Get today's reports
    todays = db_manager.get_todays_reports()
    print(f"Today's reports: {len(todays)}")
    
    for filepath, filename, model_type, size in todays:
        print(f"  {model_type}: {filename} ({size} bytes)")
    
    # Test email logging
    db_manager.log_email(
        "Daily Trading Reports",
        ["test@example.com"],
        len(todays),
        "SUCCESS"
    )
    
    # Test status
    updated, status = db_manager.get_database_status()
    print(f"Database status: {status} (Updated: {updated})")
    
    print("Database testing complete!")