"""
Configuration Management for Automated Trading Analysis System
Handles user preferences, risk parameters, and email settings
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class RiskParameters:
    """Risk management parameters"""
    max_position_size_percent: float = 3.0  # Max % of portfolio per position
    stop_loss_percent: float = 2.0          # Default stop loss %
    max_daily_risk_percent: float = 5.0     # Max daily portfolio risk
    max_open_positions: int = 4             # Max concurrent positions
    risk_tolerance: str = "MEDIUM"          # LOW, MEDIUM, HIGH
    
    # Position sizing rules based on confidence
    high_confidence_multiplier: float = 1.5  # Increase position size for high confidence
    low_confidence_multiplier: float = 0.5   # Reduce position size for low confidence
    min_confidence_threshold: float = 35.0   # Don't trade below this confidence
    
    # Options specific parameters
    max_options_risk_percent: float = 2.0    # Max % for options positions
    preferred_options_expiration_days: int = 30  # Preferred DTE
    max_options_expiration_days: int = 60    # Never go beyond this DTE

@dataclass
class EmailPreferences:
    """Email notification preferences"""
    enabled: bool = True
    recipient_email: str = ""
    send_daily_analysis: bool = True
    send_trade_alerts: bool = True
    send_performance_updates: bool = True
    send_error_notifications: bool = True
    
    # Timing preferences
    daily_analysis_time: str = "16:30"  # 4:30 PM ET
    summary_frequency: str = "DAILY"    # DAILY, WEEKLY, MONTHLY
    
    # Content preferences
    include_options_strategies: bool = True
    include_technical_analysis: bool = True
    include_risk_warnings: bool = True
    detail_level: str = "DETAILED"      # SUMMARY, DETAILED, FULL

@dataclass
class TradingPreferences:
    """Trading strategy preferences"""
    # Asset preferences
    enabled_assets: List[str] = None
    preferred_assets: List[str] = None
    
    # Strategy preferences
    prefer_long_positions: bool = True
    prefer_short_positions: bool = False
    use_options_strategies: bool = True
    use_swing_trading: bool = True
    use_day_trading: bool = False
    
    # Time horizon preferences
    min_hold_period_days: int = 1
    max_hold_period_days: int = 30
    preferred_time_horizon: str = "SHORT_TERM"  # SHORT_TERM, MEDIUM_TERM, LONG_TERM
    
    # Model preferences (weight different models)
    model_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.enabled_assets is None:
            self.enabled_assets = ["QQQ", "NVDA", "BTC", "SPY"]
        if self.preferred_assets is None:
            self.preferred_assets = ["QQQ", "NVDA"]
        if self.model_weights is None:
            self.model_weights = {
                "nvidia": 1.2,      # Higher weight for NVIDIA model
                "longhorn": 1.0,    # Standard weight
                "trading_signal": 1.0,
                "wishing_wealth": 0.8,  # Lower weight
                "bitcoin": 1.0,
                "algorand": 0.6     # Lowest weight
            }

@dataclass
class SystemConfiguration:
    """Overall system configuration"""
    # File paths
    reports_db_path: str = "reports_tracking.db"
    market_data_db_path: str = "market_data.db"
    config_file_path: str = "trading_config.json"
    log_file_path: str = "trading_analysis.log"
    
    # System settings
    timezone: str = "US/Eastern"
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    
    # Dashboard settings
    dashboard_url: str = "https://your-dashboard.com"
    api_timeout_seconds: int = 30
    
    # Logging settings
    log_level: str = "INFO"
    max_log_file_size_mb: int = 10

class ConfigurationManager:
    """Manages all configuration settings"""
    
    def __init__(self, config_file: str = "trading_config.json"):
        self.config_file = config_file
        self.risk_params = RiskParameters()
        self.email_prefs = EmailPreferences()
        self.trading_prefs = TradingPreferences()
        self.system_config = SystemConfiguration()
        
        # Load existing configuration
        self.load_configuration()
    
    def load_configuration(self) -> bool:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # Load each section
                if 'risk_parameters' in config_data:
                    self.risk_params = RiskParameters(**config_data['risk_parameters'])
                
                if 'email_preferences' in config_data:
                    self.email_prefs = EmailPreferences(**config_data['email_preferences'])
                
                if 'trading_preferences' in config_data:
                    self.trading_prefs = TradingPreferences(**config_data['trading_preferences'])
                
                if 'system_configuration' in config_data:
                    self.system_config = SystemConfiguration(**config_data['system_configuration'])
                
                logger.info(f"Configuration loaded from {self.config_file}")
                return True
            else:
                logger.info("No existing configuration file found. Using defaults.")
                return self.save_configuration()  # Save defaults
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False
    
    def save_configuration(self) -> bool:
        """Save configuration to file"""
        try:
            config_data = {
                'risk_parameters': asdict(self.risk_params),
                'email_preferences': asdict(self.email_prefs),
                'trading_preferences': asdict(self.trading_prefs),
                'system_configuration': asdict(self.system_config),
                'last_updated': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def get_position_size(self, confidence: float, asset: str) -> float:
        """Calculate position size based on confidence and risk parameters"""
        base_size = self.risk_params.max_position_size_percent
        
        # Adjust based on confidence
        if confidence >= 80:
            multiplier = self.risk_params.high_confidence_multiplier
        elif confidence <= 50:
            multiplier = self.risk_params.low_confidence_multiplier
        else:
            # Linear interpolation between 50-80% confidence
            multiplier = 0.5 + (confidence - 50) / 30 * (1.5 - 0.5)
        
        # Adjust based on asset preference
        if asset in self.trading_prefs.preferred_assets:
            multiplier *= 1.2
        
        # Adjust based on risk tolerance
        risk_multipliers = {"LOW": 0.7, "MEDIUM": 1.0, "HIGH": 1.3}
        multiplier *= risk_multipliers.get(self.risk_params.risk_tolerance, 1.0)
        
        position_size = base_size * multiplier
        
        # Ensure within bounds
        return min(position_size, self.risk_params.max_position_size_percent * 2)
    
    def get_stop_loss_percent(self, confidence: float, volatility: str = "MEDIUM") -> float:
        """Calculate stop loss percentage based on confidence and volatility"""
        base_stop = self.risk_params.stop_loss_percent
        
        # Adjust based on confidence
        if confidence >= 80:
            confidence_multiplier = 1.2  # Wider stops for high confidence
        elif confidence <= 50:
            confidence_multiplier = 0.8  # Tighter stops for low confidence
        else:
            confidence_multiplier = 1.0
        
        # Adjust based on volatility
        volatility_multipliers = {"LOW": 0.8, "MEDIUM": 1.0, "HIGH": 1.4}
        volatility_multiplier = volatility_multipliers.get(volatility, 1.0)
        
        return base_stop * confidence_multiplier * volatility_multiplier
    
    def should_trade_asset(self, asset: str) -> bool:
        """Check if we should trade a specific asset"""
        return asset in self.trading_prefs.enabled_assets
    
    def get_model_weight(self, model_name: str) -> float:
        """Get weight for a specific model"""
        return self.trading_prefs.model_weights.get(model_name, 1.0)
    
    def get_options_expiration_days(self, strategy_type: str = "standard") -> int:
        """Get preferred expiration days for options strategies"""
        if strategy_type == "aggressive":
            return min(self.risk_params.preferred_options_expiration_days, 21)
        elif strategy_type == "conservative":
            return min(self.risk_params.preferred_options_expiration_days * 1.5, 
                      self.risk_params.max_options_expiration_days)
        else:
            return self.risk_params.preferred_options_expiration_days
    
    def should_send_email(self, email_type: str) -> bool:
        """Check if we should send a specific type of email"""
        if not self.email_prefs.enabled:
            return False
        
        type_mapping = {
            'daily_analysis': self.email_prefs.send_daily_analysis,
            'trade_alerts': self.email_prefs.send_trade_alerts,
            'performance_updates': self.email_prefs.send_performance_updates,
            'error_notifications': self.email_prefs.send_error_notifications
        }
        
        return type_mapping.get(email_type, False)
    
    def get_email_recipient(self) -> str:
        """Get email recipient address"""
        # First try configuration
        if self.email_prefs.recipient_email:
            return self.email_prefs.recipient_email
        
        # Fall back to environment variables
        return os.getenv('TRADING_ANALYSIS_EMAIL', 
                        os.getenv('EMAIL_RECIPIENTS', '').split(',')[0] if os.getenv('EMAIL_RECIPIENTS') else '')
    
    def update_risk_parameters(self, **kwargs) -> bool:
        """Update risk parameters"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.risk_params, key):
                    setattr(self.risk_params, key, value)
            
            return self.save_configuration()
        except Exception as e:
            logger.error(f"Error updating risk parameters: {e}")
            return False
    
    def update_email_preferences(self, **kwargs) -> bool:
        """Update email preferences"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.email_prefs, key):
                    setattr(self.email_prefs, key, value)
            
            return self.save_configuration()
        except Exception as e:
            logger.error(f"Error updating email preferences: {e}")
            return False
    
    def update_trading_preferences(self, **kwargs) -> bool:
        """Update trading preferences"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.trading_prefs, key):
                    setattr(self.trading_prefs, key, value)
            
            return self.save_configuration()
        except Exception as e:
            logger.error(f"Error updating trading preferences: {e}")
            return False
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration"""
        return {
            'risk_tolerance': self.risk_params.risk_tolerance,
            'max_position_size': f"{self.risk_params.max_position_size_percent}%",
            'stop_loss': f"{self.risk_params.stop_loss_percent}%",
            'max_positions': self.risk_params.max_open_positions,
            'email_enabled': self.email_prefs.enabled,
            'daily_analysis_enabled': self.email_prefs.send_daily_analysis,
            'enabled_assets': ', '.join(self.trading_prefs.enabled_assets),
            'preferred_assets': ', '.join(self.trading_prefs.preferred_assets),
            'options_enabled': self.trading_prefs.use_options_strategies,
            'recipient_email': self.get_email_recipient(),
            'config_file': self.config_file,
            'last_updated': datetime.fromtimestamp(os.path.getmtime(self.config_file)).strftime('%Y-%m-%d %H:%M:%S') if os.path.exists(self.config_file) else 'Never'
        }
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate risk parameters
        if self.risk_params.max_position_size_percent <= 0 or self.risk_params.max_position_size_percent > 100:
            issues.append("Max position size must be between 0 and 100%")
        
        if self.risk_params.stop_loss_percent <= 0 or self.risk_params.stop_loss_percent > 50:
            issues.append("Stop loss must be between 0 and 50%")
        
        if self.risk_params.max_open_positions <= 0:
            issues.append("Max open positions must be greater than 0")
        
        # Validate email preferences
        if self.email_prefs.enabled and not self.get_email_recipient():
            issues.append("Email enabled but no recipient email configured")
        
        # Validate trading preferences
        if not self.trading_prefs.enabled_assets:
            issues.append("No assets enabled for trading")
        
        # Validate model weights
        total_weight = sum(self.trading_prefs.model_weights.values())
        if total_weight <= 0:
            issues.append("Total model weights must be greater than 0")
        
        return issues

# Global configuration instance
config_manager = ConfigurationManager()

def get_config() -> ConfigurationManager:
    """Get the global configuration manager instance"""
    return config_manager

def setup_configuration(config_file: str = None, **kwargs) -> ConfigurationManager:
    """Setup configuration with custom parameters"""
    global config_manager
    
    if config_file:
        config_manager = ConfigurationManager(config_file)
    
    # Update any provided parameters
    if kwargs:
        for section, params in kwargs.items():
            if section == 'risk' and isinstance(params, dict):
                config_manager.update_risk_parameters(**params)
            elif section == 'email' and isinstance(params, dict):
                config_manager.update_email_preferences(**params)
            elif section == 'trading' and isinstance(params, dict):
                config_manager.update_trading_preferences(**params)
    
    return config_manager

def create_sample_configuration():
    """Create a sample configuration file for reference"""
    sample_config = {
        "risk_parameters": {
            "max_position_size_percent": 3.0,
            "stop_loss_percent": 2.5,
            "max_daily_risk_percent": 5.0,
            "max_open_positions": 4,
            "risk_tolerance": "MEDIUM",
            "high_confidence_multiplier": 1.5,
            "low_confidence_multiplier": 0.5,
            "min_confidence_threshold": 40.0,
            "max_options_risk_percent": 2.0,
            "preferred_options_expiration_days": 30,
            "max_options_expiration_days": 60
        },
        "email_preferences": {
            "enabled": True,
            "recipient_email": "your-email@example.com",
            "send_daily_analysis": True,
            "send_trade_alerts": True,
            "send_performance_updates": True,
            "send_error_notifications": True,
            "daily_analysis_time": "16:30",
            "summary_frequency": "DAILY",
            "include_options_strategies": True,
            "include_technical_analysis": True,
            "include_risk_warnings": True,
            "detail_level": "DETAILED"
        },
        "trading_preferences": {
            "enabled_assets": ["QQQ", "NVDA", "BTC", "SPY"],
            "preferred_assets": ["QQQ", "NVDA"],
            "prefer_long_positions": True,
            "prefer_short_positions": False,
            "use_options_strategies": True,
            "use_swing_trading": True,
            "use_day_trading": False,
            "min_hold_period_days": 1,
            "max_hold_period_days": 30,
            "preferred_time_horizon": "SHORT_TERM",
            "model_weights": {
                "nvidia": 1.2,
                "longhorn": 1.0,
                "trading_signal": 1.0,
                "wishing_wealth": 0.8,
                "bitcoin": 1.0,
                "algorand": 0.6
            }
        },
        "system_configuration": {
            "reports_db_path": "reports_tracking.db",
            "market_data_db_path": "market_data.db",
            "config_file_path": "trading_config.json",
            "log_file_path": "trading_analysis.log",
            "timezone": "US/Eastern",
            "dashboard_url": "https://your-dashboard.com",
            "log_level": "INFO"
        }
    }
    
    with open('sample_trading_config.json', 'w', encoding='utf-8') as f:
        json.dump(sample_config, f, indent=2)
    
    print("✅ Sample configuration created: sample_trading_config.json")

if __name__ == "__main__":
    # Create sample configuration
    create_sample_configuration()
    
    # Test configuration manager
    config = get_config()
    print(f"Configuration summary:")
    summary = config.get_configuration_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Validate configuration
    issues = config.validate_configuration()
    if issues:
        print(f"\nConfiguration issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\n✅ Configuration is valid")