"""
Comprehensive validation module for trading reports system.
Handles data validation, cleaning, and normalization for all report types.
"""

import re
import logging
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

logger = logging.getLogger("trading_reports.validation")

@dataclass
class ValidationResult:
    """Result of data validation with comprehensive error tracking"""
    is_valid: bool
    cleaned_data: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    field_status: Dict[str, str] = field(default_factory=dict)
    
    def has_errors(self) -> bool:
        """Check if validation has any errors"""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if validation has any warnings"""
        return len(self.warnings) > 0
    
    def get_error_messages(self) -> List[str]:
        """Get all error messages"""
        return self.errors.copy()
    
    def get_warning_messages(self) -> List[str]:
        """Get all warning messages"""
        return self.warnings.copy()
    
    def add_error(self, message: str, field: str = None) -> None:
        """Add an error message"""
        self.errors.append(message)
        if field:
            self.field_status[field] = "error"
    
    def add_warning(self, message: str, field: str = None) -> None:
        """Add a warning message"""
        self.warnings.append(message)
        if field:
            self.field_status[field] = "warning"
    
    def set_field_valid(self, field: str) -> None:
        """Mark a field as valid"""
        self.field_status[field] = "valid"

class DataValidator:
    """Main data validation class"""
    
    VALID_ACTIONS = {'BUY', 'SELL', 'HOLD', 'NEUTRAL', 'STRONG_BUY', 'STRONG_SELL'}
    VALID_SIGNALS = {'BUY', 'SELL', 'HOLD', 'NEUTRAL', 'BULLISH', 'BEARISH'}
    
    REQUIRED_FIELDS = {
        'longhorn': ['report_date', 'current_price', 'suggested_action'],
        'trading_signal': ['report_date', 'current_price', 'signal', 'suggested_action'],
        'algorand': ['report_date', 'current_price', 'suggested_action'],
        'bitcoin': ['report_date', 'current_price', 'suggested_action'],
        'wishing_well': ['report_date', 'current_price', 'suggested_action'],
        'nvidia': ['report_date', 'current_price', 'suggested_action'],
        'default': ['report_date', 'current_price', 'suggested_action']
    }
    
    OPTIONAL_FIELDS = [
        'price_prediction', 'expected_return', 'trading_accuracy', 
        'signal', 'gmi_score', 'confidence_level'
    ]
    
    def __init__(self):
        self.logger = logging.getLogger("trading_reports.validation")
    
    def validate_date(self, date_value: Any, field_name: str = "date") -> Tuple[bool, str, Optional[str]]:
        """Validate date field. Returns: (is_valid, cleaned_value, error_message)"""
        if not date_value or date_value in ['N/A', '', None]:
            return False, 'N/A', f"{field_name} is missing or empty"
        
        date_str = str(date_value).strip()
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{2}/\d{2}/\d{4})',
            r'(\d{4}/\d{2}/\d{2})',
            r'(\d{8})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                date_part = match.group(1)
                try:
                    if '-' in date_part:
                        parsed_date = datetime.strptime(date_part, '%Y-%m-%d')
                    elif '/' in date_part and date_part.count('/') == 2:
                        if date_part.startswith('20'):
                            parsed_date = datetime.strptime(date_part, '%Y/%m/%d')
                        else:
                            parsed_date = datetime.strptime(date_part, '%m/%d/%Y')
                    elif len(date_part) == 8:
                        parsed_date = datetime.strptime(date_part, '%Y%m%d')
                    else:
                        continue
                    
                    standard_date = parsed_date.strftime('%Y-%m-%d')
                    return True, standard_date, None
                    
                except ValueError:
                    continue
        
        return False, date_str, f"Invalid date format in {field_name}: {date_str}"
    
    def validate_price(self, price_value: Any, field_name: str = "price") -> Tuple[bool, str, Optional[str]]:
        """Validate price field. Returns: (is_valid, cleaned_value, error_message)"""
        if not price_value or price_value in ['N/A', '', None]:
            return False, 'N/A', f"{field_name} is missing or empty"
        
        price_str = str(price_value).strip()
        cleaned_price = price_str.replace('$', '').replace(',', '').replace(' ', '')
        
        if cleaned_price.startswith('-'):
            return False, price_str, f"Negative price not allowed in {field_name}: {price_str}"
        
        try:
            price_float = float(cleaned_price)
            
            if price_float < 0:
                return False, price_str, f"Negative price in {field_name}: {price_str}"
            
            if price_float > 1000000:
                return False, price_str, f"Unreasonably high price in {field_name}: {price_str}"
            
            if price_float >= 1000:
                formatted_price = f"${price_float:,.2f}"
            else:
                formatted_price = f"${price_float:.4f}" if price_float < 1 else f"${price_float:.2f}"
            
            return True, formatted_price, None
            
        except (ValueError, TypeError):
            return False, price_str, f"Invalid price format in {field_name}: {price_str}"
    
    def validate_percentage(self, percentage_value: Any, field_name: str = "percentage") -> Tuple[bool, str, Optional[str]]:
        """Validate percentage field. Returns: (is_valid, cleaned_value, error_message)"""
        if not percentage_value or percentage_value in ['N/A', '', None]:
            return False, 'N/A', f"{field_name} is missing or empty"
        
        pct_str = str(percentage_value).strip()
        cleaned_pct = pct_str.replace('%', '').replace(' ', '')
        
        if cleaned_pct.startswith('+'):
            cleaned_pct = cleaned_pct[1:]
        
        try:
            pct_float = float(cleaned_pct)
            
            if 'accuracy' in field_name.lower() or 'hit_rate' in field_name.lower():
                if pct_float < 0 or pct_float > 100:
                    return False, pct_str, f"Accuracy percentage out of range (0-100%) in {field_name}: {pct_str}"
            elif 'return' in field_name.lower():
                if abs(pct_float) > 1000:
                    return False, pct_str, f"Return percentage seems unreasonable in {field_name}: {pct_str}"
            
            if pct_float >= 0:
                formatted_pct = f"{pct_float:.2f}%" if 'return' in field_name.lower() else f"{pct_float:.1f}%"
            else:
                formatted_pct = f"{pct_float:.2f}%"
            
            return True, formatted_pct, None
            
        except (ValueError, TypeError):
            return False, pct_str, f"Invalid percentage format in {field_name}: {pct_str}"
    
    def validate_action(self, action_value: Any, field_name: str = "action") -> Tuple[bool, str, Optional[str]]:
        """Validate trading action field. Returns: (is_valid, cleaned_value, error_message)"""
        if not action_value or action_value in ['N/A', '', None]:
            return False, 'N/A', f"{field_name} is missing or empty"
        
        action_str = str(action_value).strip().upper()
        
        action_mapping = {
            'B': 'BUY',
            'S': 'SELL',
            'H': 'HOLD',
            'N': 'NEUTRAL',
            'STRONG BUY': 'STRONG_BUY',
            'STRONG SELL': 'STRONG_SELL',
            'STRONGBUY': 'STRONG_BUY',
            'STRONGSELL': 'STRONG_SELL',
        }
        
        if action_str in action_mapping:
            action_str = action_mapping[action_str]
        
        if action_str in self.VALID_ACTIONS:
            return True, action_str, None
        else:
            valid_actions_str = ', '.join(self.VALID_ACTIONS)
            return False, str(action_value), f"Invalid trading action in {field_name}: {action_value}. Valid actions: {valid_actions_str}"
    
    def validate_signal(self, signal_value: Any, field_name: str = "signal") -> Tuple[bool, str, Optional[str]]:
        """Validate trading signal field. Returns: (is_valid, cleaned_value, error_message)"""
        if not signal_value or signal_value in ['N/A', '', None]:
            return False, 'N/A', f"{field_name} is missing or empty"
        
        signal_str = str(signal_value).strip().upper()
        
        if signal_str in self.VALID_SIGNALS:
            return True, signal_str, None
        else:
            valid_signals_str = ', '.join(self.VALID_SIGNALS)
            return False, str(signal_value), f"Invalid signal in {field_name}: {signal_value}. Valid signals: {valid_signals_str}"

def validate_report_data(data: Dict[str, Any], report_type: str = "default") -> ValidationResult:
    """Main validation function for report data"""
    validator = DataValidator()
    result = ValidationResult(is_valid=True, cleaned_data={})
    
    if not data:
        result.add_error("No data provided for validation")
        result.is_valid = False
        return result
    
    result.cleaned_data = data.copy()
    required_fields = validator.REQUIRED_FIELDS.get(report_type, validator.REQUIRED_FIELDS['default'])
    
    for field in required_fields:
        if field not in data:
            result.add_error(f"Required field '{field}' is missing", field)
            result.is_valid = False
            continue
        
        if field == 'report_date':
            is_valid, cleaned_value, error = validator.validate_date(data[field], field)
            if is_valid:
                result.cleaned_data[field] = cleaned_value
                result.set_field_valid(field)
            else:
                if error:
                    result.add_error(error, field)
                result.is_valid = False
        
        elif 'price' in field:
            is_valid, cleaned_value, error = validator.validate_price(data[field], field)
            if is_valid:
                result.cleaned_data[field] = cleaned_value
                result.set_field_valid(field)
            else:
                if error:
                    result.add_warning(error, field)
        
        elif field in ['suggested_action']:
            is_valid, cleaned_value, error = validator.validate_action(data[field], field)
            if is_valid:
                result.cleaned_data[field] = cleaned_value
                result.set_field_valid(field)
            else:
                if error:
                    result.add_warning(error, field)
        
        elif field == 'signal':
            is_valid, cleaned_value, error = validator.validate_signal(data[field], field)
            if is_valid:
                result.cleaned_data[field] = cleaned_value
                result.set_field_valid(field)
            else:
                if error:
                    result.add_warning(error, field)
    
    for field in validator.OPTIONAL_FIELDS:
        if field in data and data[field] not in ['N/A', '', None]:
            
            if 'price' in field:
                is_valid, cleaned_value, error = validator.validate_price(data[field], field)
                if is_valid:
                    result.cleaned_data[field] = cleaned_value
                    result.set_field_valid(field)
                else:
                    if error:
                        result.add_warning(error, field)
            
            elif 'return' in field or 'accuracy' in field:
                is_valid, cleaned_value, error = validator.validate_percentage(data[field], field)
                if is_valid:
                    result.cleaned_data[field] = cleaned_value
                    result.set_field_valid(field)
                else:
                    if error:
                        result.add_warning(error, field)
            
            elif field == 'signal':
                is_valid, cleaned_value, error = validator.validate_signal(data[field], field)
                if is_valid:
                    result.cleaned_data[field] = cleaned_value
                    result.set_field_valid(field)
                else:
                    if error:
                        result.add_warning(error, field)
    
    perform_cross_validation(result, report_type)
    return result

def perform_cross_validation(result: ValidationResult, report_type: str) -> None:
    """Perform cross-field validation checks"""
    data = result.cleaned_data
    
    if all(field in data and data[field] != 'N/A' for field in ['current_price', 'price_prediction', 'expected_return']):
        try:
            current = float(data['current_price'].replace('$', '').replace(',', ''))
            predicted = float(data['price_prediction'].replace('$', '').replace(',', ''))
            expected_return_str = data['expected_return'].replace('%', '').replace('+', '')
            expected_return = float(expected_return_str)
            
            actual_return = ((predicted / current) - 1) * 100
            
            if abs(actual_return - expected_return) > 0.1:
                warning_msg = f"Price prediction ({data['price_prediction']}) and expected return ({data['expected_return']}) are inconsistent"
                result.add_warning(warning_msg)
        
        except (ValueError, TypeError, ZeroDivisionError):
            pass
    
    if 'signal' in data and 'suggested_action' in data:
        if data['signal'] != 'N/A' and data['suggested_action'] != 'N/A':
            signal = data['signal']
            action = data['suggested_action']
            
            if signal in ['BUY', 'BULLISH'] and action not in ['BUY', 'STRONG_BUY']:
                result.add_warning(f"Signal '{signal}' inconsistent with action '{action}'")
            elif signal in ['SELL', 'BEARISH'] and action not in ['SELL', 'STRONG_SELL']:
                result.add_warning(f"Signal '{signal}' inconsistent with action '{action}'")

def clean_report_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and normalize report data"""
    if not data:
        return {}
    
    cleaned = {}
    
    for key, value in data.items():
        if value is None:
            cleaned[key] = 'N/A'
        elif isinstance(value, str):
            cleaned_value = value.strip()
            if cleaned_value == '':
                cleaned[key] = 'N/A'
            else:
                cleaned[key] = cleaned_value
        elif isinstance(value, (int, float)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value).strip() if str(value).strip() else 'N/A'
    
    return cleaned

def validate_batch_data(data_list: List[Dict[str, Any]], report_type: str = "default") -> List[ValidationResult]:
    """Validate multiple report data entries"""
    results = []
    for i, data in enumerate(data_list):
        try:
            result = validate_report_data(data, report_type)
            results.append(result)
        except Exception as e:
            error_result = ValidationResult(
                is_valid=False,
                cleaned_data=data.copy() if data else {},
                errors=[f"Validation failed for entry {i}: {str(e)}"]
            )
            results.append(error_result)
            logger.error(f"Validation failed for entry {i}: {e}")
    
    return results

def get_validation_summary(results: List[ValidationResult]) -> Dict[str, Any]:
    """Get summary statistics for validation results"""
    total = len(results)
    valid = sum(1 for r in results if r.is_valid)
    invalid = total - valid
    total_errors = sum(len(r.errors) for r in results)
    total_warnings = sum(len(r.warnings) for r in results)
    
    return {
        'total_records': total,
        'valid_records': valid,
        'invalid_records': invalid,
        'success_rate': (valid / total * 100) if total > 0 else 0,
        'total_errors': total_errors,
        'total_warnings': total_warnings,
        'avg_errors_per_record': total_errors / total if total > 0 else 0,
        'avg_warnings_per_record': total_warnings / total if total > 0 else 0
    }

def validate_price_format(price_str: str) -> bool:
    """Legacy function for price validation"""
    validator = DataValidator()
    is_valid, _, _ = validator.validate_price(price_str)
    return is_valid

def validate_percentage_format(pct_str: str) -> bool:
    """Legacy function for percentage validation"""
    validator = DataValidator()
    is_valid, _, _ = validator.validate_percentage(pct_str)
    return is_valid

def normalize_action(action: str) -> str:
    """Legacy function for action normalization"""
    validator = DataValidator()
    is_valid, cleaned_value, _ = validator.validate_action(action)
    return cleaned_value if is_valid else action