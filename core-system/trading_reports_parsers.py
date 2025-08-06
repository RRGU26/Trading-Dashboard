"""
Trading Reports Parser System
Provides standalone functions and classes that match existing test script expectations
VERIFIED: 2025-07-19 - All syntax correct, enhanced Wishing Wealth parsing
UPDATED: Value cleaning to fix "VERY" and other parsing artifacts
UPDATED: Search both OneDrive Desktop and regular Desktop locations
"""

import re
import os
import glob
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================================================
# VALUE CLEANING FUNCTION (fixes "VERY" and other parsing artifacts)
# ========================================================================

def clean_extracted_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """Clean up extracted values to remove parsing artifacts"""
    
    cleaned_data = data.copy()
    
    # Clean confidence values
    if 'confidence' in cleaned_data:
        confidence = str(cleaned_data['confidence']).strip()
        
        # Convert "VERY" to meaningful confidence levels
        if confidence.upper() == 'VERY':
            cleaned_data['confidence'] = '95%'  # Very high confidence
        elif confidence.upper() == 'VERY HIGH':
            cleaned_data['confidence'] = '95%'
        elif confidence.upper() == 'VERY LOW':
            cleaned_data['confidence'] = '15%'
        elif confidence.upper() == 'HIGH':
            cleaned_data['confidence'] = '85%'
        elif confidence.upper() == 'MEDIUM':
            cleaned_data['confidence'] = '65%'
        elif confidence.upper() == 'LOW':
            cleaned_data['confidence'] = '35%'
        elif 'CONFIDENCE' in confidence.upper():
            # Remove extra "CONFIDENCE" text
            cleaned_data['confidence'] = confidence.replace('CONFIDENCE', '').strip()
            if cleaned_data['confidence'].upper() == 'VERY':
                cleaned_data['confidence'] = '95%'
    
    # Clean suggested_action values
    if 'suggested_action' in cleaned_data:
        action = str(cleaned_data['suggested_action']).strip()
        
        # Remove extra text artifacts
        if 'CONFIDENCE' in action.upper():
            action = action.replace('CONFIDENCE', '').strip()
        if 'CURRENT PRICE' in action.upper():
            action = action.replace('CURRENT PRICE', '').strip()
        
        # Standardize action values
        action_upper = action.upper()
        if action_upper in ['BUY', 'STRONG BUY']:
            cleaned_data['suggested_action'] = 'BUY'
        elif action_upper in ['SELL', 'STRONG SELL']:
            cleaned_data['suggested_action'] = 'SELL'
        elif action_upper in ['HOLD', 'NEUTRAL']:
            cleaned_data['suggested_action'] = 'HOLD'
        else:
            cleaned_data['suggested_action'] = action.upper()
    
    # Clean risk_level values
    if 'risk_level' in cleaned_data:
        risk = str(cleaned_data['risk_level']).strip()
        
        if risk.upper() == 'VERY':
            cleaned_data['risk_level'] = 'VERY HIGH'
        elif risk.upper() == 'VERY HIGH':
            cleaned_data['risk_level'] = 'VERY HIGH'
        elif risk.upper() == 'VERY LOW':
            cleaned_data['risk_level'] = 'VERY LOW'
        elif risk.upper() in ['HIGH', 'MEDIUM', 'LOW']:
            cleaned_data['risk_level'] = risk.upper()
    
    # Clean signal values
    if 'signal' in cleaned_data:
        signal = str(cleaned_data['signal']).strip()
        signal_upper = signal.upper()
        
        if signal_upper in ['BUY', 'STRONG BUY', 'BULLISH']:
            cleaned_data['signal'] = 'BUY'
        elif signal_upper in ['SELL', 'STRONG SELL', 'BEARISH']:
            cleaned_data['signal'] = 'SELL'
        elif signal_upper in ['HOLD', 'NEUTRAL']:
            cleaned_data['signal'] = 'HOLD'
    
    return cleaned_data

# ========================================================================
# INDIVIDUAL PARSER FUNCTIONS (updated with value cleaning)
# ========================================================================

def parse_algorand_report(content: str) -> Dict[str, Any]:
    """Parse Algorand Model V2 Integrated Reports"""
    patterns = {
        'current_price': r'Current Price:\s*\$?([\d.]+)',
        'predicted_price': r'Predicted Price.*?:\s*\$?([\d.]+)',
        'predicted_1_day_price': r'Predicted Price \(1d\):\s*\$?([\d.]+)',
        'expected_return': r'Expected Return:\s*([-+]?[\d.]+)%',
        'ensemble_1_day_return': r'Ensemble.*?1.*?day.*?return:\s*([-+]?[\d.]+)%',
        'confidence': r'Confidence Level:\s*([\d.]+)%?',
        'signal': r'Signal:\s*(\w+(?:\s+\w+)*)',
        'suggested_action': r'Suggested Action:\s*(\w+(?:\s+\w+)*)',
        'market_regime': r'Market Regime:\s*(\w+(?:_\w+)*)',
        'r_squared': r'R.Squared:\s*([\d.]+)',
        'mae': r'MAE:\s*([\d.]+)',
        'direction_accuracy': r'Direction Accuracy:\s*([\d.]+)%?',
        'report_timestamp': r'(\d{4}-\d{2}-\d{2})',
    }
    
    extracted_data = {}
    
    for field_name, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            
            # Convert numeric fields
            numeric_fields = [
                'current_price', 'predicted_price', 'predicted_1_day_price',
                'confidence', 'r_squared', 'mae', 'direction_accuracy'
            ]
            
            if field_name in numeric_fields:
                try:
                    extracted_data[field_name] = float(value)
                except ValueError:
                    extracted_data[field_name] = value
            elif field_name == 'report_timestamp':
                try:
                    extracted_data[field_name] = datetime.strptime(value, '%Y-%m-%d').isoformat()
                except ValueError:
                    extracted_data[field_name] = value
            else:
                extracted_data[field_name] = value.upper()
    
    # APPLY VALUE CLEANING
    extracted_data = clean_extracted_values(extracted_data)
    
    return extracted_data

def parse_nvidia_report(content: str) -> Dict[str, Any]:
    """Parse NVIDIA Bull Momentum Reports"""
    patterns = {
        'current_price': r'Current Price:\s*\$?([\d.]+)',
        'confidence': r'Confidence:\s*(\w+(?:\s+\w+)*?)(?:\n|$)',
        'signal': r'Signal:\s*(\w+(?:\s+\w+)*?)(?:\n|$)',
        'suggested_action': r'Suggested Action:\s*(\w+(?:\s+\w+)*?)(?:\n|$)',
        'predicted_1_day_price': r'Predicted.*?1.*?day.*?Price:\s*\$?([\d.]+)',
        'predicted_1_day_return': r'Predicted.*?1.*?day.*?Return:\s*([-+]?[\d.]+)%',
        'predicted_5_day_price': r'Predicted.*?5.*?day.*?Price:\s*\$?([\d.]+)', 
        'predicted_5_day_return': r'Predicted.*?5.*?day.*?Return:\s*([-+]?[\d.]+)%',
        'report_timestamp': r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})',
    }
    
    extracted_data = {}
    
    for field_name, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            
            numeric_fields = [
                'current_price', 'predicted_1_day_price', 'predicted_5_day_price'
            ]
            
            if field_name in numeric_fields:
                try:
                    extracted_data[field_name] = float(value)
                except ValueError:
                    extracted_data[field_name] = value
            elif field_name == 'report_timestamp':
                try:
                    extracted_data[field_name] = datetime.fromisoformat(value.replace('T', ' ')).isoformat()
                except ValueError:
                    extracted_data[field_name] = value
            else:
                extracted_data[field_name] = value.upper()
    
    # APPLY VALUE CLEANING
    extracted_data = clean_extracted_values(extracted_data)
    
    return extracted_data

def parse_bitcoin_report(content: str) -> Dict[str, Any]:
    """Parse Bitcoin Prediction Reports"""
    patterns = {
        'current_price': r'Current Price:\s*\$?([\d,]+\.?\d*)',
        'signal': r'Signal:\s*(\w+)',
        'suggested_action': r'Suggested Action:\s*(\w+(?:\s+\w+)*)',
        'predicted_1_day_price': r'Predicted.*?1.*?day.*?Price:\s*\$?([\d,]+\.?\d*)',
        'predicted_1_day_return': r'Predicted.*?1.*?day.*?Return:\s*([-+]?[\d.]+)%',
        'predicted_3_day_price': r'Predicted.*?3.*?day.*?Price:\s*\$?([\d,]+\.?\d*)',
        'predicted_3_day_return': r'Predicted.*?3.*?day.*?Return:\s*([-+]?[\d.]+)%',
        'predicted_7_day_price': r'Predicted.*?7.*?day.*?Price:\s*\$?([\d,]+\.?\d*)',
        'predicted_7_day_return': r'Predicted.*?7.*?day.*?Return:\s*([-+]?[\d.]+)%',
        'report_timestamp': r'(\d{4}-\d{2}-\d{2})',
    }
    
    extracted_data = {}
    
    for field_name, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            
            numeric_fields = [
                'current_price', 'predicted_1_day_price', 'predicted_3_day_price', 'predicted_7_day_price'
            ]
            
            if field_name in numeric_fields:
                try:
                    # Remove commas and convert to float
                    cleaned_value = value.replace(',', '')
                    extracted_data[field_name] = float(cleaned_value)
                except ValueError:
                    extracted_data[field_name] = value
            elif field_name == 'report_timestamp':
                try:
                    extracted_data[field_name] = datetime.strptime(value, '%Y-%m-%d').isoformat()
                except ValueError:
                    extracted_data[field_name] = value
            else:
                extracted_data[field_name] = value.upper()
    
    # APPLY VALUE CLEANING
    extracted_data = clean_extracted_values(extracted_data)
    
    return extracted_data

def parse_wishing_wealth_report(content: str) -> Dict[str, Any]:
    """Parse Wishing Wealth QQQ Trading Signal reports - ENHANCED VERSION"""
    patterns = {
        # Basic signal information
        'signal': r'Signal:\s*(\w+)',
        'suggested_action': r'Suggested Action:\s*(\w+)',
        'current_price': r'Current Price:\s*\$?([\d,]+\.?\d*)',
        
        # Scoring and trend information  
        'gmi_score': r'GMI Score:\s*(\d+)\s*/\s*(\d+)',
        'gmi_score_color': r'GMI Score:.*?\((\w+)\)',
        'qqq_trend': r'QQQ Trend:\s*([^\n\r]+)',
        'win_rate': r'Win Rate:\s*([\d.]+)%',
        
        # ETF Strategy
        'recommended_etf': r'Recommended ETF:\s*(\w+)',
        'strategy': r'Strategy:\s*([^\n\r]+)',
        
        # Performance metrics - QQQ Buy and Hold
        'qqq_total_return': r'QQQ Buy and Hold Strategy:.*?Total Return:\s*([\d.-]+)%',
        'qqq_annualized_return': r'QQQ Buy and Hold Strategy:.*?Annualized Return:\s*([\d.-]+)%',
        'qqq_sharpe_ratio': r'QQQ Buy and Hold Strategy:.*?Sharpe Ratio:\s*([\d.-]+)',
        'qqq_max_drawdown': r'QQQ Buy and Hold Strategy:.*?Maximum Drawdown:\s*-([\d.]+)%',
        'qqq_win_rate_strategy': r'QQQ Buy and Hold Strategy:.*?Win Rate:\s*([\d.]+)%',
        
        # Performance metrics - QQQ Timing Strategy
        'timing_total_return': r'(?:QQQ Timing Strategy|Basic QQQ Timing Strategy).*?Total Return:\s*([-\d.]+)%',
        'timing_annualized_return': r'(?:QQQ Timing Strategy|Basic QQQ Timing Strategy).*?Annualized Return:\s*([-\d.]+)%',
        'timing_sharpe_ratio': r'(?:QQQ Timing Strategy|Basic QQQ Timing Strategy).*?Sharpe Ratio:\s*([\d.-]+)',
        'timing_max_drawdown': r'(?:QQQ Timing Strategy|Basic QQQ Timing Strategy).*?Maximum Drawdown:\s*-([\d.]+)%',
        'timing_win_rate': r'(?:QQQ Timing Strategy|Basic QQQ Timing Strategy).*?Win Rate:\s*([\d.]+)%',
        'timing_number_of_trades': r'(?:QQQ Timing Strategy|Basic QQQ Timing Strategy).*?Number of Trades:\s*(\d+)',
        
        # Performance metrics - Leveraged Strategy
        'leveraged_total_return': r'Leveraged TQQQ/SQQQ Strategy:.*?Total Return:\s*([-\d.]+)%',
        'leveraged_annualized_return': r'Leveraged TQQQ/SQQQ Strategy:.*?Annualized Return:\s*([-\d.]+)%',
        'leveraged_sharpe_ratio': r'Leveraged TQQQ/SQQQ Strategy:.*?Sharpe Ratio:\s*([\d.-]+)',
        'leveraged_max_drawdown': r'Leveraged TQQQ/SQQQ Strategy:.*?Maximum Drawdown:\s*-([\d.]+)%',
        'leveraged_win_rate': r'Leveraged TQQQ/SQQQ Strategy:.*?Win Rate:\s*([\d.]+)%',
        'leveraged_number_of_trades': r'Leveraged TQQQ/SQQQ Strategy:.*?Number of Trades:\s*(\d+)',
        
        # Performance comparison
        'strategy_vs_qqq_outperformance': r'Strategy vs QQQ Buy-Hold:\s*([-+]?[\d.]+)%\s*outperformance',
        'leveraged_vs_basic_return': r'Leveraged vs Basic Strategy:\s*([-+]?[\d.]+)%\s*additional return',
        
        # Technical Analysis
        'moving_averages': r'[•·]\s*Moving Averages:\s*(\w+)',
        'rsi': r'[•·]\s*RSI:\s*([\d.]+)\s*-\s*(\w+)',
        'rsi_signal': r'[•·]\s*RSI:.*?\((\w+)\)',
        'macd': r'[•·]\s*MACD:\s*(\w+)',
        'bollinger_bands': r'[•·]\s*Bollinger Bands:\s*([^\n\r]+)',
        'vix': r'[•·]\s*VIX:\s*([\d.]+)\s*-\s*(\w+)',
        'vix_signal': r'[•·]\s*VIX:.*?\((\w+)\)',
        
        # Other fields
        'backtest_period': r'Backtest Period:\s*([^\n\r]+)',
        'generated_date': r'Generated on:\s*(\d{4}-\d{2}-\d{2})',
        'price_prediction': r'PRICE PREDICTION:\s*([^\n\r]+)'
    }
    
    extracted_data = {}
    
    for field_name, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if match:
            if field_name == 'gmi_score':
                # Special handling for GMI score - create fraction string
                extracted_data['gmi_score'] = f"{match.group(1)}/{match.group(2)}"
                extracted_data['gmi_score_numerator'] = float(match.group(1))
                extracted_data['gmi_score_denominator'] = float(match.group(2))
            elif field_name in ['rsi', 'vix']:
                # Extract both value and status for RSI and VIX
                extracted_data[f'{field_name}_value'] = float(match.group(1))
                extracted_data[f'{field_name}_status'] = match.group(2)
            else:
                value = match.group(1).strip()
                
                # Convert numeric fields
                numeric_fields = [
                    'current_price', 'win_rate', 
                    'qqq_total_return', 'qqq_annualized_return', 'qqq_sharpe_ratio', 'qqq_max_drawdown', 'qqq_win_rate_strategy',
                    'timing_total_return', 'timing_annualized_return', 'timing_sharpe_ratio', 'timing_max_drawdown', 'timing_win_rate', 'timing_number_of_trades',
                    'leveraged_total_return', 'leveraged_annualized_return', 'leveraged_sharpe_ratio', 'leveraged_max_drawdown', 'leveraged_win_rate', 'leveraged_number_of_trades',
                    'strategy_vs_qqq_outperformance', 'leveraged_vs_basic_return'
                ]
                
                if field_name in numeric_fields:
                    try:
                        cleaned_value = value.replace(',', '')
                        extracted_data[field_name] = float(cleaned_value)
                    except ValueError:
                        extracted_data[field_name] = value
                else:
                    extracted_data[field_name] = value
    
    # Add computed fields
    if 'gmi_score_numerator' in extracted_data and 'gmi_score_denominator' in extracted_data:
        gmi_percentage = (extracted_data['gmi_score_numerator'] / extracted_data['gmi_score_denominator']) * 100
        extracted_data['gmi_score_percentage'] = gmi_percentage
        
        # Add confidence based on GMI score
        if gmi_percentage >= 83.33:  # 5/6 or 6/6
            extracted_data['confidence'] = 'High'
        elif gmi_percentage >= 66.67:  # 4/6
            extracted_data['confidence'] = 'Medium'
        elif gmi_percentage >= 50.0:   # 3/6
            extracted_data['confidence'] = 'Low'
        else:
            extracted_data['confidence'] = 'Very Low'
    
    # Add report timestamp
    if 'generated_date' in extracted_data:
        try:
            date_str = extracted_data['generated_date']
            extracted_data['report_timestamp'] = datetime.strptime(date_str, '%Y-%m-%d').isoformat()
        except ValueError:
            pass
    
    # Normalize signals
    for field in ['signal', 'suggested_action']:
        if field in extracted_data:
            extracted_data[field] = extracted_data[field].upper()
    
    # APPLY VALUE CLEANING
    extracted_data = clean_extracted_values(extracted_data)
    
    return extracted_data

def parse_trading_signal_report(content: str) -> Dict[str, Any]:
    """Parse QQQ Trading Signal reports"""
    patterns = {
        'signal': r'Signal:\s*(\w+)',
        'confidence': r'Confidence:\s*([\d.]+)%?',
        'current_price': r'Current Price:\s*\$?([\d.]+)',
        'suggested_action': r'Suggested Action:\s*(\w+(?:\s+\w+)*)',
        'price_target': r'Price Target:\s*\$?([\d.]+)',
        'stop_loss': r'Stop Loss:\s*\$?([\d.]+)',
        'report_timestamp': r'(\d{4}-\d{2}-\d{2})',
    }
    
    extracted_data = {}
    
    for field_name, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            
            numeric_fields = ['confidence', 'current_price', 'price_target', 'stop_loss']
            
            if field_name in numeric_fields:
                try:
                    extracted_data[field_name] = float(value)
                except ValueError:
                    extracted_data[field_name] = value
            elif field_name == 'report_timestamp':
                try:
                    extracted_data[field_name] = datetime.strptime(value, '%Y-%m-%d').isoformat()
                except ValueError:
                    extracted_data[field_name] = value
            else:
                extracted_data[field_name] = value.upper()
    
    # APPLY VALUE CLEANING
    extracted_data = clean_extracted_values(extracted_data)
    
    return extracted_data

def parse_longhorn_report(content: str) -> Dict[str, Any]:
    """Parse QQQ Long Bull Reports"""
    patterns = {
        'current_price': r'Current Price:\s*\$?([\d.]+)',
        'signal': r'Signal:\s*(\w+)',
        'suggested_action': r'Suggested Action:\s*(\w+(?:\s+\w+)*)',
        'confidence': r'Confidence:\s*(\w+)',
        'target_price': r'Target Price:\s*\$?([\d.]+)',
        'stop_loss': r'Stop Loss:\s*\$?([\d.]+)',
        'risk_level': r'Risk Level:\s*(\w+)',
        'report_timestamp': r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})',
    }
    
    extracted_data = {}
    
    for field_name, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            
            numeric_fields = ['current_price', 'target_price', 'stop_loss']
            
            if field_name in numeric_fields:
                try:
                    extracted_data[field_name] = float(value)
                except ValueError:
                    extracted_data[field_name] = value
            elif field_name == 'report_timestamp':
                try:
                    extracted_data[field_name] = datetime.fromisoformat(value.replace('T', ' ')).isoformat()
                except ValueError:
                    extracted_data[field_name] = value
            else:
                extracted_data[field_name] = value.upper()
    
    # APPLY VALUE CLEANING
    extracted_data = clean_extracted_values(extracted_data)
    
    return extracted_data

# ========================================================================
# ENHANCED PARSER CLASSES (for backward compatibility)
# ========================================================================

class EnhancedAlgorandReportParser:
    """Enhanced Algorand parser class"""
    
    def __init__(self, filepath=None, *args, **kwargs):
        """Initialize the parser - accepts filepath or any arguments for compatibility"""
        self.filepath = filepath
        self.content = None
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    self.content = file.read()
            except Exception as e:
                logger.error(f"Failed to load file {filepath}: {str(e)}")
    
    def parse(self, content: str = None) -> Dict[str, Any]:
        """Parse Algorand report content"""
        content_to_parse = content or self.content
        if not content_to_parse:
            raise ValueError("No content provided for parsing")
        return parse_algorand_report(content_to_parse)
    
    @staticmethod
    def parse_static(content: str) -> Dict[str, Any]:
        """Static method for parsing"""
        return parse_algorand_report(content)

class EnhancedNVIDIAReportParser:
    """Enhanced NVIDIA parser class"""
    
    def __init__(self, filepath=None, *args, **kwargs):
        """Initialize the parser - accepts filepath or any arguments for compatibility"""
        self.filepath = filepath
        self.content = None
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    self.content = file.read()
            except Exception as e:
                logger.error(f"Failed to load file {filepath}: {str(e)}")
    
    def parse(self, content: str = None) -> Dict[str, Any]:
        """Parse NVIDIA report content"""
        content_to_parse = content or self.content
        if not content_to_parse:
            raise ValueError("No content provided for parsing")
        return parse_nvidia_report(content_to_parse)
    
    @staticmethod
    def parse_static(content: str) -> Dict[str, Any]:
        """Static method for parsing"""
        return parse_nvidia_report(content)

class EnhancedBitcoinReportParser:
    """Enhanced Bitcoin parser class"""
    
    def __init__(self, filepath=None, *args, **kwargs):
        """Initialize the parser - accepts filepath or any arguments for compatibility"""
        self.filepath = filepath
        self.content = None
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    self.content = file.read()
            except Exception as e:
                logger.error(f"Failed to load file {filepath}: {str(e)}")
    
    def parse(self, content: str = None) -> Dict[str, Any]:
        """Parse Bitcoin report content"""
        content_to_parse = content or self.content
        if not content_to_parse:
            raise ValueError("No content provided for parsing")
        return parse_bitcoin_report(content_to_parse)
    
    @staticmethod
    def parse_static(content: str) -> Dict[str, Any]:
        """Static method for parsing"""
        return parse_bitcoin_report(content)

class EnhancedWishingWealthReportParser:
    """Enhanced Wishing Wealth parser class - ENHANCED VERSION"""
    
    def __init__(self, filepath=None, *args, **kwargs):
        """Initialize the parser - accepts filepath or any arguments for compatibility"""
        self.filepath = filepath
        self.content = None
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    self.content = file.read()
            except Exception as e:
                logger.error(f"Failed to load file {filepath}: {str(e)}")
    
    def parse(self, content: str = None) -> Dict[str, Any]:
        """Parse Wishing Wealth report content"""
        content_to_parse = content or self.content
        if not content_to_parse:
            raise ValueError("No content provided for parsing")
        return parse_wishing_wealth_report(content_to_parse)
    
    @staticmethod
    def parse_static(content: str) -> Dict[str, Any]:
        """Static method for parsing"""
        return parse_wishing_wealth_report(content)

class EnhancedTradingSignalReportParser:
    """Enhanced Trading Signal parser class"""
    
    def __init__(self, filepath=None, *args, **kwargs):
        """Initialize the parser - accepts filepath or any arguments for compatibility"""
        self.filepath = filepath
        self.content = None
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    self.content = file.read()
            except Exception as e:
                logger.error(f"Failed to load file {filepath}: {str(e)}")
    
    def parse(self, content: str = None) -> Dict[str, Any]:
        """Parse Trading Signal report content"""
        content_to_parse = content or self.content
        if not content_to_parse:
            raise ValueError("No content provided for parsing")
        return parse_trading_signal_report(content_to_parse)
    
    @staticmethod
    def parse_static(content: str) -> Dict[str, Any]:
        """Static method for parsing"""
        return parse_trading_signal_report(content)

class EnhancedLonghornReportParser:
    """Enhanced Longhorn parser class"""
    
    def __init__(self, filepath=None, *args, **kwargs):
        """Initialize the parser - accepts filepath or any arguments for compatibility"""
        self.filepath = filepath
        self.content = None
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    self.content = file.read()
            except Exception as e:
                logger.error(f"Failed to load file {filepath}: {str(e)}")
    
    def parse(self, content: str = None) -> Dict[str, Any]:
        """Parse Longhorn report content"""
        content_to_parse = content or self.content
        if not content_to_parse:
            raise ValueError("No content provided for parsing")
        return parse_longhorn_report(content_to_parse)
    
    @staticmethod
    def parse_static(content: str) -> Dict[str, Any]:
        """Static method for parsing"""
        return parse_longhorn_report(content)

# ========================================================================
# MAIN PARSING FUNCTIONS (UPDATED FOR DUAL DIRECTORY SEARCH)
# ========================================================================

def parse_report_file(filepath: str, report_type: str) -> Dict[str, Any]:
    """
    Parse a single report file based on its type
    
    Args:
        filepath: Path to the report file
        report_type: Type of report ('algorand', 'nvidia', 'bitcoin', etc.)
        
    Returns:
        Dictionary containing extracted data fields
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        
        parser_functions = {
            'algorand': parse_algorand_report,
            'nvidia': parse_nvidia_report,
            'bitcoin': parse_bitcoin_report,
            'wishing_wealth': parse_wishing_wealth_report,
            'trading_signal': parse_trading_signal_report,
            'longhorn': parse_longhorn_report
        }
        
        if report_type in parser_functions:
            result = parser_functions[report_type](content)
            result['report_type'] = report_type
            result['file_path'] = filepath
            result['file_name'] = os.path.basename(filepath)
            return result
        else:
            logger.error(f"Unknown report type: {report_type}")
            return {}
            
    except Exception as e:
        logger.error(f"Failed to parse {filepath}: {str(e)}")
        return {}

def find_report_files(directories: List[str]) -> Dict[str, List[str]]:
    """Find all report files in the specified directories - UPDATED FOR MULTIPLE DIRECTORIES"""
    file_patterns = {
        'algorand': 'Algorand_Model_V2_Integrated_Report_*.txt',
        'nvidia': 'NVIDIA_Bull_Momentum_Report_*.txt',
        'bitcoin': 'Bitcoin_Prediction_Report*_*.txt',  # UPDATED to catch suffixes like _FIXED
        'wishing_wealth': 'WishingWealthQQQ_signal.txt',
        'trading_signal': 'QQQ_Trading_Signal*.txt',
        'longhorn': 'QQQ_Long_Bull_Report_*.txt'
    }
    
    found_files = {}
    
    for report_type, pattern in file_patterns.items():
        all_files_for_type = []
        
        # Search in all directories
        for directory in directories:
            if os.path.exists(directory):
                search_path = os.path.join(directory, pattern)
                files = glob.glob(search_path)
                all_files_for_type.extend(files)
                if files:
                    logger.debug(f"Found {len(files)} {report_type} files in {directory}")
        
        # Sort by modification time (newest first) and remove duplicates
        unique_files = list(set(all_files_for_type))
        found_files[report_type] = sorted(unique_files, key=os.path.getmtime, reverse=True)
    
    return found_files

def parse_all_reports(directory: str = None, latest_only: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Parse all reports in the specified directories - UPDATED FOR DUAL DIRECTORY SEARCH
    
    Args:
        directory: Primary directory containing report files (defaults to ~/OneDrive/Desktop)
        latest_only: If True, parse only the most recent file for each report type
        
    Returns:
        Dictionary with report type as key and parsed data as value
    """
    
    # UPDATED: Search in both OneDrive Desktop and regular Desktop
    if directory is None:
        directories = [
            os.path.expanduser("~/OneDrive/Desktop"),
            os.path.expanduser("~/Desktop")
        ]
    else:
        # If a specific directory is provided, also search regular Desktop as backup
        directories = [
            directory,
            os.path.expanduser("~/Desktop")
        ]
    
    # Remove duplicates and non-existent directories
    directories = [d for d in set(directories) if os.path.exists(d)]
    
    logger.info(f"Searching for report files in: {', '.join(directories)}")
    
    # Find all files
    all_files = find_report_files(directories)
    
    # Count and log file statistics
    file_counts = {report_type: len(files) for report_type, files in all_files.items()}
    
    for report_type, files in all_files.items():
        if files:
            latest_file = os.path.basename(files[0])
            file_dir = os.path.dirname(files[0])
            logger.info(f"Found {len(files)} {report_type} files, latest: {latest_file} in {file_dir}")
    
    total_files = sum(file_counts.values())
    logger.info(f"Total files found: {total_files}")
    
    if latest_only:
        files_to_parse = len([count for count in file_counts.values() if count > 0])
        logger.info(f"Parsing {files_to_parse} latest report files")
    
    # Parse files
    parser_functions = {
        'algorand': parse_algorand_report,
        'nvidia': parse_nvidia_report,
        'bitcoin': parse_bitcoin_report,
        'wishing_wealth': parse_wishing_wealth_report,
        'trading_signal': parse_trading_signal_report,
        'longhorn': parse_longhorn_report
    }
    
    results = {}
    
    for report_type, files in all_files.items():
        if not files:
            logger.warning(f"No {report_type} files found")
            continue
        
        if latest_only:
            files_to_process = [files[0]]  # Just the latest
        else:
            files_to_process = files
        
        for filepath in files_to_process:
            logger.info(f"Parsing {report_type}: {os.path.basename(filepath)}")
            
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                if report_type in parser_functions:
                    parsed_data = parser_functions[report_type](content)
                    
                    if parsed_data:
                        meaningful_fields = len([v for k, v in parsed_data.items() 
                                               if v is not None and v != ''])
                        logger.info(f"Successfully parsed {report_type}: {meaningful_fields} fields extracted")
                        results[report_type] = parsed_data
                    else:
                        logger.warning(f"No data extracted from {report_type}")
                
            except Exception as e:
                logger.error(f"Failed to parse {filepath}: {str(e)}")
    
    return results

# ========================================================================
# UTILITY FUNCTIONS
# ========================================================================

def get_latest_trading_signals(directory: str = None) -> Dict[str, str]:
    """Get just the trading signals from all latest reports"""
    all_data = parse_all_reports(directory, latest_only=True)
    
    signals = {}
    for report_type, data in all_data.items():
        if 'signal' in data:
            signals[report_type] = data['signal']
    
    return signals

def print_parsing_summary(results: Dict[str, Dict[str, Any]]):
    """Print a formatted summary of parsing results"""
    print(f"\n🎯 PARSING SUMMARY")
    print("=" * 60)
    
    if not results:
        print("❌ No reports parsed successfully")
        return
    
    print(f"✅ Parsed {len(results)} report types\n")
    
    for report_type, data in results.items():
        if not data:
            print(f"📊 {report_type.upper()}:")
            print(f"  ❌ No data extracted\n")
            continue
        
        # Count meaningful fields
        meaningful_fields = len([v for k, v in data.items() if v is not None and v != ''])
        
        print(f"📊 {report_type.upper()}:")
        print(f"  📈 {meaningful_fields} meaningful fields extracted")
        
        # Show key fields
        key_fields = ['current_price', 'signal', 'confidence', 'suggested_action']
        for field in key_fields:
            if field in data and data[field] is not None:
                print(f"    ✅ {field}: {data[field]}")
        
        print()

# ========================================================================
# BACKWARDS COMPATIBILITY ALIASES
# ========================================================================

# Create aliases for any other expected names
WishingWealthParser = EnhancedWishingWealthReportParser
parse_wishing_wealth = parse_wishing_wealth_report

# Additional aliases for different naming conventions
EnhancedWishingWealthParser = EnhancedWishingWealthReportParser
EnhancedNvidiaReportParser = EnhancedNVIDIAReportParser
NVIDIAReportParser = EnhancedNVIDIAReportParser
AlgorandReportParser = EnhancedAlgorandReportParser
BitcoinReportParser = EnhancedBitcoinReportParser
TradingSignalReportParser = EnhancedTradingSignalReportParser
LonghornReportParser = EnhancedLonghornReportParser

if __name__ == "__main__":
    # Test all parsers
    print("🧪 TESTING UPDATED TRADING REPORT PARSERS")
    print("=" * 60)
    
    results = parse_all_reports(latest_only=True)
    print_parsing_summary(results)
    
    # Test signals
    signals = get_latest_trading_signals()
    print("🎯 LATEST TRADING SIGNALS")
    print("=" * 30)
    for report_type, signal in signals.items():
        print(f"📊 {report_type.upper()}: {signal}")