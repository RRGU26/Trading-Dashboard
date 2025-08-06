"""
Configuration management for trading reports system.
Centralized configuration with environment variable support.
FIXED: Synchronized with trading_reports_parsers.py patterns
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import logging

@dataclass
class EmailConfig:
    """Email configuration settings"""
    sender: str = 'RRGU26@gmail.com'
    recipients: List[str] = field(default_factory=lambda: [
        'RRGU26@gmail.com', 
        'timbarney62@gmail.com',
        'rebeccalynnrosenthal@gmail.com', 
        'samkest419@gmail.com',
        'georgelaffey@gmail.com', 
        'gmrosenthal1@gmail.com', 
        'david.worldco@gmail.com'
    ])
    smtp_server: str = 'smtp.gmail.com'
    smtp_port: int = 587
    smtp_user: str = 'RRGU26@gmail.com'
    password_env_var: str = 'GMAIL_APP_PASSWORD'
    
    @property
    def password(self) -> Optional[str]:
        """Get password from environment variable"""
        return os.environ.get(self.password_env_var)
    
    @property
    def is_configured(self) -> bool:
        """Check if email is properly configured"""
        return bool(self.password and self.password.strip())

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    db_name: str = "models_dashboard.db"
    backup_enabled: bool = True
    backup_retention_days: int = 30
    connection_timeout: int = 30
    
    @property
    def db_path(self) -> Path:
        """Get full database path"""
        script_dir = Path(__file__).parent
        return script_dir / self.db_name
    
    @property
    def backup_dir(self) -> Path:
        """Get backup directory path"""
        script_dir = Path(__file__).parent
        backup_dir = script_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        return backup_dir

@dataclass
class ReportConfig:
    """Report generation configuration"""
    symbols_to_update: List[str] = field(default_factory=lambda: [
        'QQQ', 'BTC-USD', 'ALGO-USD', 'VIX'
    ])
    horizon_days: int = 3
    dashboard_url: str = "https://rrgu26-trading-dashboard-dashboard-hogil1.streamlit.app/"
    
    # FIXED: Report file patterns synchronized with trading_reports_parsers.py
    # These EXACTLY match the patterns in find_report_files() function
    report_patterns: Dict[str, List[str]] = field(default_factory=lambda: {
        'longhorn': [
            "QQQ_Long_Bull_Report_*.txt"  # EXACT pattern from parsers
        ],
        'trading_signal': [
            "QQQ_Trading_Signal*.txt"     # EXACT pattern from parsers
        ],
        'algorand': [
            "Algorand_Model_V2_Integrated_Report_*.txt"  # EXACT pattern from parsers
        ],
        'bitcoin': [
            "Bitcoin_Prediction_Report_*.txt"  # EXACT pattern from parsers
        ],
        'wishing_wealth': [  # FIXED: was 'wishing_well', now matches parser key
            "WishingWealthQQQ_signal.txt"      # EXACT pattern from parsers
        ],
        'nvidia': [
            "NVIDIA_Bull_Momentum_Report_*.txt"  # EXACT pattern from parsers
        ]
    })

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    log_file: str = "trading_reports.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    def setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        from logging.handlers import RotatingFileHandler
        
        # Create logger
        logger = logging.getLogger("trading_reports")
        logger.setLevel(getattr(logging, self.level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            self.log_file, 
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(self.format)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

@dataclass
class ValidationConfig:
    """Data validation configuration"""
    max_price: float = 1000000.0  # Maximum reasonable price
    min_price: float = 0.01       # Minimum reasonable price
    max_return_percent: float = 100.0  # Maximum reasonable return %
    min_return_percent: float = -100.0  # Minimum reasonable return %
    valid_actions: List[str] = field(default_factory=lambda: ['BUY', 'SELL', 'HOLD'])

class TradingReportsConfig:
    """Main configuration class that combines all config sections"""
    
    def __init__(self):
        self.email = EmailConfig()
        self.database = DatabaseConfig()
        self.reports = ReportConfig()
        self.logging = LoggingConfig()
        self.validation = ValidationConfig()
        
        # Setup logging
        self.logger = self.logging.setup_logging()
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check email configuration
        if not self.email.is_configured:
            issues.append("Email password not configured (GMAIL_APP_PASSWORD environment variable)")
        
        # Check database path is writable
        try:
            self.database.db_path.parent.mkdir(exist_ok=True)
            test_file = self.database.db_path.parent / "test_write.tmp"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            issues.append(f"Database directory not writable: {e}")
        
        # Check dashboard URL is reachable (optional)
        if not self.reports.dashboard_url.startswith(('http://', 'https://')):
            issues.append("Dashboard URL should start with http:// or https://")
        
        return issues
    
    def get_desktop_path(self) -> Path:
        """Get the correct desktop path based on OS"""
        import platform
        
        system = platform.system()
        
        if system == "Windows":
            # Try OneDrive desktop first
            user_profile = os.environ.get('USERPROFILE', '')
            onedrive_desktop = Path(user_profile) / 'OneDrive' / 'Desktop'
            if onedrive_desktop.exists():
                return onedrive_desktop
            return Path(user_profile) / 'Desktop'
        else:
            desktop = Path.home() / 'Desktop'
            desktop.mkdir(exist_ok=True)
            return desktop
    
    def get_script_dir(self) -> Path:
        """Get the directory where the script is located"""
        return Path(__file__).parent
    
    def get_search_paths(self, report_type: str) -> List[Path]:
        """Get all search paths for a specific report type"""
        if report_type not in self.reports.report_patterns:
            return []
        
        patterns = self.reports.report_patterns[report_type]
        desktop_path = self.get_desktop_path()
        script_dir = self.get_script_dir()
        
        search_paths = []
        for pattern in patterns:
            # Add desktop path variations
            search_paths.append(desktop_path / pattern)
            # Add script directory variations
            search_paths.append(script_dir / pattern)
        
        return search_paths

# Global configuration instance
config = TradingReportsConfig()

# Convenience functions for backward compatibility
def get_desktop_path():
    return config.get_desktop_path()

def get_script_dir():
    return config.get_script_dir()

def setup_logging():
    return config.logger