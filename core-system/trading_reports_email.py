"""
Complete Email System for Trading Reports
Includes comprehensive formatting and email sending functionality
Updated: 2025-07-19 - Fixed syntax errors and added Wishing Wealth support
"""

import re
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

class ComprehensiveEmailFormatter:
    """Enhanced email formatter that shows ALL available metrics including Wishing Wealth"""
    
    def __init__(self):
        self.metric_categories = self._define_metric_categories()
    
    def _define_metric_categories(self) -> Dict[str, Dict[str, Any]]:
        """Define how to categorize and format different types of metrics"""
        return {
            'price_data': {
                'title': '💰 Price Information',
                'fields': [
                    'current_price', 'predicted_price', 'price_prediction',
                    '1_day_price_prediction', '3_day_price_prediction', 
                    '5_day_price_prediction', '7_day_price_prediction',
                    'ensemble_1_day_price', 'ensemble_3_day_price', 'ensemble_5_day_price',
                    'target_price', 'price_target', 'predicted_1_day_price', 'predicted_3_day_price',
                    'predicted_5_day_price', 'predicted_7_day_price'
                ],
                'color': '#28a745',
                'format': 'currency'
            },
            'returns_performance': {
                'title': '📈 Returns & Performance',
                'fields': [
                    'expected_return', '1_day_expected_return', '3_day_expected_return',
                    '5_day_expected_return', '7_day_expected_return', 'price_change',
                    'ensemble_1_day_return', 'ensemble_3_day_return', 'ensemble_5_day_return',
                    'predicted_1_day_return', 'predicted_3_day_return', 'predicted_5_day_return',
                    'predicted_7_day_return'
                ],
                'color': '#17a2b8',
                'format': 'percentage'
            },
            'wishing_wealth_strategy': {
                'title': '🔮 Wishing Wealth Strategy',
                'fields': [
                    'gmi_score', 'gmi_score_percentage', 'qqq_trend', 'recommended_etf', 
                    'timing_total_return', 'leveraged_total_return', 'qqq_total_return',
                    'strategy_vs_qqq_outperformance', 'leveraged_vs_basic_return',
                    'win_rate', 'qqq_win_rate_strategy', 'timing_win_rate', 'leveraged_win_rate',
                    'timing_sharpe_ratio', 'leveraged_sharpe_ratio', 'qqq_sharpe_ratio',
                    'timing_max_drawdown', 'leveraged_max_drawdown', 'qqq_max_drawdown',
                    'timing_number_of_trades', 'leveraged_number_of_trades', 'strategy'
                ],
                'color': '#9c27b0',
                'format': 'strategy'
            },
            'technical_indicators': {
                'title': '⚡ Technical Indicators',
                'fields': [
                    'rsi', 'rsi_value', 'rsi_status', 'rsi_signal', 'macd', 'macd_signal',
                    'bollinger_upper', 'bollinger_lower', 'bollinger_bands',
                    'vix', 'vix_value', 'vix_status', 'vix_signal',
                    'fear_greed_index', 'volatility', 'moving_averages'
                ],
                'color': '#fd7e14',
                'format': 'technical'
            },
            'moving_averages': {
                'title': '📊 Moving Averages',
                'fields': [
                    '5_day_ma', '10_day_ma', '20_day_ma', '50_day_ma', '200_day_ma',
                    'sma5', 'sma10', 'sma20', 'sma50', 'ema5', 'ema10', 'ema20'
                ],
                'color': '#6f42c1',
                'format': 'currency'
            },
            'accuracy_metrics': {
                'title': '🎯 Model Accuracy',
                'fields': [
                    'trading_accuracy', 'accuracy', 'hit_rate', 'direction_accuracy',
                    'confidence', 'prediction_confidence', 'r_squared', 'mae'
                ],
                'color': '#20c997',
                'format': 'percentage'
            },
            'trading_signals': {
                'title': '🚦 Trading Signals',
                'fields': [
                    'signal', 'suggested_action', 'recommendation', 
                    'market_sentiment', 'sentiment', 'market_regime'
                ],
                'color': '#dc3545',
                'format': 'action'
            },
            'risk_metrics': {
                'title': '⚠️ Risk Analysis',
                'fields': [
                    'max_drawdown', 'sharpe_ratio', 'profit_factor', 
                    'support_resistance_levels', 'risk_level', 'stop_loss'
                ],
                'color': '#6c757d',
                'format': 'raw'
            },
            'market_data': {
                'title': '🌐 Market Context',
                'fields': [
                    'volume', 'market_cap', 'trading_volume', 'average_volume'
                ],
                'color': '#495057',
                'format': 'raw'
            }
        }
    
    def generate_comprehensive_html_table(self, title: str, date: str, data: Dict[str, Any]) -> str:
        """Generate comprehensive HTML table showing ALL available metrics"""
        
        # Filter out metadata fields
        display_data = {k: v for k, v in data.items() 
                      if k not in ['report_date', 'file_path', 'report_type'] 
                      and v and v != "N/A" and v != ""}
        
        if not display_data:
            return f"""<div class="model-section">
                <div class="model-header">
                    <h3 class="model-title">{title} - {date}</h3>
                </div>
                <div class="model-content">
                    <p style="color: #6c757d; font-style: italic; text-align: center; padding: 20px;">No data available</p>
                </div>
            </div>"""
        
        # Build categorized rows
        all_rows = ""
        used_fields = set()
        
        # Process each category
        for category_key, category_info in self.metric_categories.items():
            category_fields = category_info['fields']
            category_data = {}
            
            # Find matching fields for this category
            for field in category_fields:
                if field in display_data:
                    category_data[field] = display_data[field]
                    used_fields.add(field)
            
            # Add any fields that contain category keywords
            for field, value in display_data.items():
                if field not in used_fields:
                    if self._field_matches_category(field, category_key):
                        category_data[field] = value
                        used_fields.add(field)
            
            # Generate rows for this category if we have data
            if category_data:
                all_rows += self._generate_category_rows(
                    category_info['title'], 
                    category_data, 
                    category_info['color'],
                    category_info['format']
                )
        
        # Add any remaining fields that didn't fit into categories
        remaining_data = {k: v for k, v in display_data.items() if k not in used_fields}
        if remaining_data:
            all_rows += self._generate_category_rows(
                "📋 Additional Metrics", 
                remaining_data, 
                "#6c757d",
                "raw"
            )
        
        return f"""<div class="model-section">
            <div class="model-header">
                <h3 class="model-title">{title} - {date}</h3>
            </div>
            <div class="model-content">
                <table class="responsive-table">
                    {all_rows}
                </table>
            </div>
        </div>"""
    
    def _field_matches_category(self, field: str, category: str) -> bool:
        """Check if a field belongs to a category based on keywords"""
        field_lower = field.lower()
        
        if category == 'moving_averages':
            return any(keyword in field_lower for keyword in ['ma', 'moving', 'average', 'sma', 'ema'])
        elif category == 'price_data':
            return any(keyword in field_lower for keyword in ['price', 'target', 'level'])
        elif category == 'returns_performance':
            return any(keyword in field_lower for keyword in ['return', 'change', 'performance'])
        elif category == 'wishing_wealth_strategy':
            return any(keyword in field_lower for keyword in [
                'gmi', 'timing', 'leveraged', 'strategy', 'qqq_trend', 
                'recommended_etf', 'outperformance', 'tqqq', 'sqqq'
            ])
        elif category == 'technical_indicators':
            return any(keyword in field_lower for keyword in ['rsi', 'macd', 'bollinger', 'vix'])
        elif category == 'accuracy_metrics':
            return any(keyword in field_lower for keyword in ['accuracy', 'hit', 'win', 'confidence'])
        elif category == 'trading_signals':
            return any(keyword in field_lower for keyword in ['signal', 'action', 'recommendation'])
        elif category == 'market_data':
            return any(keyword in field_lower for keyword in ['volume', 'cap', 'market'])
        elif category == 'risk_metrics':
            return any(keyword in field_lower for keyword in ['risk', 'drawdown', 'sharpe', 'volatility'])
        
        return False
    
    def _generate_category_rows(self, category_title: str, category_data: Dict[str, Any], 
                               color: str, format_type: str) -> str:
        """Generate HTML rows for a category of metrics"""
        if not category_data:
            return ""
        
        # Category header row
        rows = f"""<tr>
            <td colspan="2" style="background-color: {color}; color: white; padding: 8px; border: 1px solid #ddd; font-weight: bold; text-align: center;">
                {category_title}
            </td>
        </tr>"""
        
        # Sort fields for better display
        sorted_items = self._sort_category_items(list(category_data.items()), format_type)
        
        # Data rows
        for field, value in sorted_items:
            display_name = self._format_field_name(field)
            formatted_value = self._format_value(value, format_type)
            value_color = self._get_value_color(value, format_type)
            
            rows += f"""<tr>
                <th style="background-color: #f8f8f8; font-weight: bold; color: #555; width: 40%; padding: 8px; border: 1px solid #ddd; text-align: left;">
                    {display_name}
                </th>
                <td style="width: 60%; padding: 8px; border: 1px solid #ddd; text-align: left; color: {value_color}; font-weight: {'bold' if format_type == 'action' else 'normal'};">
                    {formatted_value}
                </td>
            </tr>"""
        
        return rows
    
    def _sort_category_items(self, items: List[Tuple[str, Any]], format_type: str) -> List[Tuple[str, Any]]:
        """Sort items within a category for optimal display"""
        if format_type == 'action':
            # Put most important trading signals first
            priority_order = ['suggested_action', 'signal', 'recommendation', 'market_sentiment']
            sorted_items = []
            
            # Add priority items first
            for priority_field in priority_order:
                for field, value in items:
                    if field == priority_field:
                        sorted_items.append((field, value))
                        break
            
            # Add remaining items
            for field, value in items:
                if field not in [item[0] for item in sorted_items]:
                    sorted_items.append((field, value))
            
            return sorted_items
        
        elif format_type == 'strategy':
            # Sort strategy fields logically
            priority_order = [
                'gmi_score', 'qqq_trend', 'recommended_etf', 'signal', 
                'timing_total_return', 'leveraged_total_return', 'qqq_total_return',
                'strategy_vs_qqq_outperformance', 'win_rate'
            ]
            sorted_items = []
            
            # Add priority items first
            for priority_field in priority_order:
                for field, value in items:
                    if field == priority_field:
                        sorted_items.append((field, value))
                        break
            
            # Add remaining items
            for field, value in items:
                if field not in [item[0] for item in sorted_items]:
                    sorted_items.append((field, value))
            
            return sorted_items
        
        elif format_type == 'currency':
            # Sort moving averages by day count
            def ma_sort_key(item):
                field = item[0]
                # Extract number from field name
                numbers = re.findall(r'\d+', field)
                return int(numbers[0]) if numbers else 999
            
            if any('ma' in item[0].lower() for item in items):
                return sorted(items, key=ma_sort_key)
        
        # Default: sort alphabetically
        return sorted(items, key=lambda x: x[0])
    
    def _format_field_name(self, field: str) -> str:
        """Convert field names to readable labels"""
        # Handle special cases
        replacements = {
            'rsi': 'RSI',
            'macd': 'MACD', 
            'vix_level': 'VIX Level',
            'vix_value': 'VIX Value',
            'vix_status': 'VIX Status',
            'rsi_value': 'RSI Value',
            'rsi_status': 'RSI Status',
            'fear_greed_index': 'Fear & Greed Index',
            'market_cap': 'Market Cap',
            'sharpe_ratio': 'Sharpe Ratio',
            'max_drawdown': 'Max Drawdown',
            'profit_factor': 'Profit Factor',
            'gmi_score': 'GMI Score',
            'qqq_trend': 'QQQ Trend',
            'recommended_etf': 'Recommended ETF',
            'timing_total_return': 'Strategy Total Return',
            'leveraged_total_return': 'Leveraged Strategy Return',
            'qqq_total_return': 'QQQ Buy & Hold Return',
            'strategy_vs_qqq_outperformance': 'Strategy vs QQQ Outperformance',
            'direction_accuracy': 'Direction Accuracy',
            'r_squared': 'R-Squared',
            'mae': 'Mean Absolute Error'
        }
        
        if field in replacements:
            return replacements[field]
        
        # Convert snake_case to Title Case
        return field.replace('_', ' ').title()
    
    def _format_value(self, value: str, format_type: str) -> str:
        """Format values based on their type"""
        if not value or value == "N/A":
            return "N/A"
        
        if format_type == 'currency':
            # Ensure currency values start with $
            if not str(value).startswith('$'):
                # Try to extract numeric value and format as currency
                numeric_match = re.search(r'([\d,.]+)', str(value))
                if numeric_match:
                    return f"${numeric_match.group(1)}"
            return str(value)
        
        elif format_type == 'percentage':
            # Ensure percentage values end with %
            if not str(value).endswith('%'):
                return f"{value}%"
            return str(value)
        
        elif format_type == 'strategy':
            # Special formatting for strategy metrics
            str_value = str(value).lower()
            if 'return' in str_value or 'outperformance' in str_value:
                if not str(value).endswith('%'):
                    return f"{value}%"
            elif 'gmi_score' in str_value:
                return str(value)  # Keep GMI score as-is (e.g., "5/6")
            elif 'etf' in str_value or value in ['TQQQ', 'SQQQ', 'QQQ']:
                return str(value).upper()  # ETF symbols in caps
            return str(value)
        
        elif format_type == 'technical':
            # Technical indicator formatting
            if 'rsi' in str(value).lower() or 'vix' in str(value).lower():
                return str(value)
            return str(value)
        
        elif format_type == 'action':
            # Capitalize and clean action values
            return str(value).upper().strip()
        
        else:  # raw
            return str(value)
    
    def _get_value_color(self, value: str, format_type: str) -> str:
        """Get appropriate color for value based on content"""
        if format_type == 'action':
            value_lower = str(value).lower()
            if any(word in value_lower for word in ['buy', 'bull', 'long']):
                return "#28a745"  # Green
            elif any(word in value_lower for word in ['sell', 'bear', 'short']):
                return "#dc3545"  # Red
            elif any(word in value_lower for word in ['hold', 'neutral']):
                return "#ffc107"  # Yellow
            else:
                return "#6c757d"  # Gray
        
        elif format_type == 'percentage' or format_type == 'strategy':
            # Color code percentages based on positive/negative
            try:
                numeric_value = float(str(value).replace('%', '').replace('+', ''))
                if numeric_value > 0:
                    return "#28a745"  # Green for positive
                elif numeric_value < 0:
                    return "#dc3545"  # Red for negative
                else:
                    return "#6c757d"  # Gray for neutral
            except ValueError:
                return "#333333"  # Default
        
        else:
            return "#333333"  # Default color
    
    def create_comprehensive_plain_text_email(self, model_reports: Dict[str, Dict[str, Any]], 
                                            dashboard_url: str, db_status_message: str, 
                                            performance_summary: str, current_date: str) -> str:
        """Create comprehensive plain text version showing ALL metrics"""
        text_content = f"Trading Models Report - {current_date}\n\n"
        
        model_names = {
            'longhorn': 'Long Bull Model',
            'qqq_master': 'QQQ Master Model',
            'trading_signal': 'QQQ Trading Signal',
            'wishing_wealth': 'Wishing Well QQQ Model',
            'nvidia': 'NVIDIA Bull Momentum Model',
            'algorand': 'Algorand Model',
            'bitcoin': 'Bitcoin Model'
        }
        
        for model_key, model_name in model_names.items():
            data = model_reports.get(model_key, {})
            if data:
                text_content += f"{model_name} - {data.get('report_date', current_date)}\n"
                text_content += "=" * (len(model_name) + 20) + "\n"
                
                # Filter out metadata
                display_data = {k: v for k, v in data.items() 
                              if k not in ['report_date', 'file_path', 'report_type'] 
                              and v and v != "N/A" and v != ""}
                
                # Group and display by categories
                for category_key, category_info in self.metric_categories.items():
                    category_data = {}
                    
                    # Find data for this category
                    for field in category_info['fields']:
                        if field in display_data:
                            category_data[field] = display_data[field]
                    
                    # Add fields that match category keywords
                    for field, value in display_data.items():
                        if field not in category_data and self._field_matches_category(field, category_key):
                            category_data[field] = value
                    
                    # Display category if we have data
                    if category_data:
                        text_content += f"\n{category_info['title']}\n"
                        text_content += "-" * len(category_info['title']) + "\n"
                        
                        for field, value in sorted(category_data.items()):
                            display_name = self._format_field_name(field)
                            formatted_value = self._format_value(value, category_info['format'])
                            text_content += f"{display_name}: {formatted_value}\n"
                
                text_content += "\n\n"
        
        # Add footer information
        text_content += "=" * 60 + "\n"
        text_content += f"View Interactive Dashboard: {dashboard_url}\n\n"
        text_content += f"Database Status: {db_status_message}\n\n"
        text_content += f"{performance_summary}\n\n"
        text_content += f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return text_content
    
    def create_comprehensive_html_email(self, model_reports: Dict[str, Dict[str, Any]], 
                                       dashboard_url: str, db_status_message: str, 
                                       performance_summary: str, current_date: str,
                                       db_updated: bool) -> str:
        """Create comprehensive HTML email showing ALL metrics"""
        
        # Status indicator for HTML
        db_status_color = "#28a745" if db_updated else "#dc3545"
        db_status_icon = "[OK]" if db_updated else "[FAIL]"
        
        # Generate comprehensive tables for each model
        model_tables = ""
        model_configs = [
            ('longhorn', 'Long Bull Model'),
            ('qqq_master', 'QQQ Master Model'),
            ('trading_signal', 'QQQ Trading Signal'),
            ('wishing_wealth', 'Wishing Well QQQ Model'),
            ('nvidia', 'NVIDIA Bull Momentum Model'),
            ('algorand', 'Algorand Model'),
            ('bitcoin', 'Bitcoin Model')
        ]
        
        for model_key, model_name in model_configs:
            data = model_reports.get(model_key, {})
            if data:
                report_date = data.get('report_date', current_date)
                model_tables += self.generate_comprehensive_html_table(model_name, report_date, data)
        
        html_body = f"""
        <!DOCTYPE html>
        <html xmlns="http://www.w3.org/1999/xhtml">
        <head>
            <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
            <meta name="x-apple-disable-message-reformatting">
            <meta name="format-detection" content="telephone=no">
            <title>📊 Trading Models Report - {current_date}</title>
            <style type="text/css">
                /* Reset styles */
                * {{ box-sizing: border-box; }}
                body, table, td, h1, h2, h3, h4, h5, h6, p, div {{ margin: 0; padding: 0; }}
                
                /* Base styles */
                body {{
                    -webkit-text-size-adjust: 100%;
                    -ms-text-size-adjust: 100%;
                    margin: 0 !important;
                    padding: 0 !important;
                    background-color: #f4f4f4;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                    font-size: 16px;
                    line-height: 1.6;
                    color: #333333;
                }}
                
                /* Container */
                .email-container {{
                    max-width: 600px;
                    margin: 0 auto;
                    background-color: #ffffff;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }}
                
                /* Header */
                .header {{
                    background: linear-gradient(135deg, #0056b3 0%, #007bff 100%);
                    color: #ffffff;
                    padding: 30px 20px;
                    text-align: center;
                }}
                
                .header h1 {{
                    font-size: 24px;
                    font-weight: 700;
                    margin-bottom: 8px;
                    text-shadow: 0 1px 2px rgba(0,0,0,0.1);
                }}
                
                .header .subtitle {{
                    font-size: 16px;
                    opacity: 0.9;
                    font-weight: 400;
                }}
                
                /* Content area */
                .content {{
                    padding: 20px;
                }}
                
                /* Model sections */
                .model-section {{
                    margin-bottom: 30px;
                    border: 1px solid #e9ecef;
                    border-radius: 8px;
                    overflow: hidden;
                }}
                
                .model-header {{
                    background-color: #f8f9fa;
                    padding: 15px 20px;
                    border-bottom: 1px solid #e9ecef;
                }}
                
                .model-title {{
                    font-size: 18px;
                    font-weight: 600;
                    color: #0056b3;
                    margin: 0;
                }}
                
                .model-content {{
                    padding: 20px;
                }}
                
                /* Responsive table */
                .responsive-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 10px 0;
                }}
                
                .responsive-table th,
                .responsive-table td {{
                    padding: 12px 8px;
                    text-align: left;
                    border-bottom: 1px solid #e9ecef;
                    font-size: 14px;
                }}
                
                .responsive-table th {{
                    background-color: #f8f9fa;
                    font-weight: 600;
                    color: #495057;
                }}
                
                /* Attachments note */
                .attachments-note {{
                    background-color: #fff3cd;
                    border: 1px solid #ffeaa7;
                    border-radius: 6px;
                    padding: 15px;
                    margin: 20px 0;
                    text-align: center;
                }}
                
                .attachments-note h3 {{
                    color: #856404;
                    font-size: 16px;
                    margin-bottom: 8px;
                }}
                
                .attachments-note p {{
                    color: #856404;
                    font-size: 14px;
                    margin: 0;
                }}
                
                /* Footer */
                .footer {{
                    background-color: #f8f9fa;
                    padding: 20px;
                    text-align: center;
                    border-top: 1px solid #e9ecef;
                }}
                
                /* Mobile styles */
                @media only screen and (max-width: 480px) {{
                    .email-container {{
                        margin: 10px;
                        border-radius: 4px;
                    }}
                    
                    .header {{
                        padding: 20px 15px;
                    }}
                    
                    .header h1 {{
                        font-size: 20px;
                    }}
                    
                    .header .subtitle {{
                        font-size: 14px;
                    }}
                    
                    .content {{
                        padding: 15px;
                    }}
                    
                    .model-content {{
                        padding: 15px;
                    }}
                    
                    .responsive-table th,
                    .responsive-table td {{
                        padding: 8px 4px;
                        font-size: 13px;
                    }}
                    
                    .attachments-note {{
                        padding: 12px;
                        margin: 15px 0;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="email-container">
                <div class="header">
                    <h1>📊 Trading Models Report</h1>
                    <div class="subtitle">{current_date} • All Models Executed Successfully</div>
                </div>
                
                <div class="attachments-note">
                    <h3>📎 Complete Reports & Analysis</h3>
                    <p>Detailed reports and options analysis are included as attachments.<br>
                    Scroll down for the summary of all model predictions.</p>
                </div>
                
                <div class="content">
                        
                        {model_tables}
                        
                        <div style="margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #e9ecef;">
                            <h3 style="color: #0056b3; font-size: 16px; margin-top: 0; margin-bottom: 15px;">Additional Resources</h3>
                            
                            <div style="margin-bottom: 15px;">
                                <a href="{dashboard_url}" style="display: inline-block; padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 4px; font-weight: bold; margin-right: 10px;">
                                    📊 Local Dashboard
                                </a>
                                <a href="https://rrgu26-trading-dashboard-cloud-dashboard-with-sync-crrtqv.streamlit.app" style="display: inline-block; padding: 10px 20px; background-color: #28a745; color: white; text-decoration: none; border-radius: 4px; font-weight: bold;">
                                    🌐 Live Cloud Dashboard
                                </a>
                            </div>
                            
                            <div style="margin-bottom: 15px; padding: 15px; background-color: #e7f3ff; border-radius: 6px; border-left: 4px solid #007bff;">
                                <h4 style="margin: 0 0 8px 0; color: #0056b3;">📱 Access Your Dashboards</h4>
                                <p style="margin: 0; font-size: 14px;">
                                    <strong>Local Dashboard:</strong> Real-time data from your system<br>
                                    <strong>Cloud Dashboard:</strong> Mobile-optimized, accessible anywhere 24/7
                                </p>
                            </div>
                            
                            <div style="padding: 10px; background-color: white; border-radius: 4px; border-left: 4px solid {db_status_color}; margin-bottom: 15px;">
                                <strong style="color: {db_status_color};">{db_status_icon} Database Status:</strong>
                                <span style="margin-left: 10px; color: #333;">{db_status_message}</span>
                            </div>
                            
                            <div style="padding: 10px; background-color: white; border-radius: 4px; border-left: 4px solid #17a2b8;">
                                <strong style="color: #17a2b8;">Performance Summary:</strong>
                                <div style="margin-top: 5px; font-size: 12px; white-space: pre-line;">{performance_summary}</div>
                            </div>
                        </div>
                </div>
                
                <div class="footer">
                    <p style="font-size: 12px; color: #6c757d; margin: 0;">
                        Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • 
                        Trading Models System v3.0
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_body


class EmailManager:
    """Handles the actual sending of emails"""
    
    def __init__(self, smtp_server: str = "smtp.gmail.com", smtp_port: int = 587):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.formatter = ComprehensiveEmailFormatter()
        self.initialized = False
        self.sender_email = None
        self.sender_password = None
        self.recipients = []
    
    def initialize(self, sender_email: str = None, sender_password: str = None, 
                  recipients: List[str] = None) -> bool:
        """Initialize email manager with credentials and recipients"""
        try:
            # Get credentials if not provided
            if not sender_email or not sender_password:
                self.sender_email, self.sender_password = get_gmail_credentials()
            else:
                self.sender_email = sender_email
                self.sender_password = sender_password
            
            # Get recipients if not provided
            if not recipients:
                self.recipients = get_recipient_list()
            else:
                self.recipients = recipients
            
            self.initialized = True
            logger.info("Email manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize email manager: {e}")
            self.initialized = False
            return False
    
    def is_initialized(self) -> bool:
        """Check if email manager is properly initialized"""
        return self.initialized and self.sender_email and self.sender_password and self.recipients
    
    def send_test_email(self) -> bool:
        """Send a test email to verify configuration"""
        if not self.initialized:
            logger.error("Email manager not initialized")
            return False
        
        try:
            test_subject = "Trading Reports System - Test Email"
            test_body = f"""
            <html>
            <body>
            <h2>Trading Reports System Test</h2>
            <p>This is a test email to verify that the email system is working correctly.</p>
            <p>Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>If you receive this email, the email configuration is working properly.</p>
            </body>
            </html>
            """
            
            return self.send_email(
                self.sender_email, self.sender_password, self.recipients,
                test_subject, test_body, "Trading Reports System Test - Email configuration working!"
            )
        except Exception as e:
            logger.error(f"Failed to send test email: {e}")
            return False
    
    def get_configuration_status(self) -> Dict[str, Any]:
        """Get current configuration status"""
        return {
            'initialized': self.initialized,
            'smtp_server': self.smtp_server,
            'smtp_port': self.smtp_port,
            'sender_configured': bool(self.sender_email),
            'recipients_configured': bool(self.recipients),
            'recipient_count': len(self.recipients) if self.recipients else 0
        }
    
    def send_email(self, sender_email: str, sender_password: str, 
                   recipient_emails: List[str], subject: str, 
                   html_body: str, plain_text_body: str = None,
                   attachments: List[str] = None) -> bool:
        """Send an email with HTML and plain text versions"""
        
        try:
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = sender_email
            message["To"] = ", ".join(recipient_emails)
            
            # Add plain text version
            if plain_text_body:
                text_part = MIMEText(plain_text_body, "plain")
                message.attach(text_part)
            
            # Add HTML version
            html_part = MIMEText(html_body, "html")
            message.attach(html_part)
            
            # Add attachments if provided
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        with open(file_path, "rb") as attachment:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(attachment.read())
                        
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {os.path.basename(file_path)}',
                        )
                        message.attach(part)
            
            # Create secure connection and send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, recipient_emails, message.as_string())
            
            logger.info(f"Email sent successfully to {len(recipient_emails)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def send_trading_report(self, sender_email: str = None, sender_password: str = None,
                           recipient_emails: List[str] = None, model_reports: Dict[str, Dict[str, Any]] = None,
                           dashboard_url: str = "https://your-dashboard.com",
                           db_status_message: str = "Database updated successfully",
                           performance_summary: str = "All models running normally",
                           attachments: List[str] = None) -> bool:
        """Send a complete trading report email"""
        
        # Use initialized credentials if available
        if self.initialized:
            sender_email = sender_email or self.sender_email
            sender_password = sender_password or self.sender_password
            recipient_emails = recipient_emails or self.recipients
        
        # Fallback to getting credentials if not provided
        if not sender_email or not sender_password:
            try:
                sender_email, sender_password = get_gmail_credentials()
            except Exception as e:
                logger.error(f"Failed to get Gmail credentials: {e}")
                return False
        
        if not recipient_emails:
            try:
                recipient_emails = get_recipient_list()
            except Exception as e:
                logger.error(f"Failed to get recipient list: {e}")
                return False
        
        if not model_reports:
            logger.warning("No model reports provided for email")
            model_reports = {}
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Generate email content
        html_body = self.formatter.create_comprehensive_html_email(
            model_reports, dashboard_url, db_status_message, 
            performance_summary, current_date, True
        )
        
        plain_text_body = self.formatter.create_comprehensive_plain_text_email(
            model_reports, dashboard_url, db_status_message, 
            performance_summary, current_date
        )
        
        subject = f"Trading Models Report - {current_date}"
        
        return self.send_email(
            sender_email, sender_password, recipient_emails,
            subject, html_body, plain_text_body, attachments
        )


# Create global email manager instance
email_manager = EmailManager()


def send_trading_reports_email(sender_email: str, sender_password: str,
                              recipient_emails: List[str], 
                              model_reports: Dict[str, Dict[str, Any]],
                              dashboard_url: str = "https://your-dashboard.com",
                              db_status_message: str = "Database updated successfully",
                              performance_summary: str = "All models running normally") -> bool:
    """Main function to send trading reports - this is what send_report.py calls"""
    
    return email_manager.send_trading_report(
        sender_email, sender_password, recipient_emails,
        model_reports, dashboard_url, db_status_message, performance_summary
    )


def get_gmail_credentials() -> tuple:
    """Get Gmail credentials from environment variables"""
    email = os.getenv('GMAIL_EMAIL')
    password = os.getenv('GMAIL_APP_PASSWORD')
    
    if not email or not password:
        # Try alternative environment variable names
        email = email or os.getenv('GMAIL_USER') or os.getenv('EMAIL_USER')
        password = password or os.getenv('GMAIL_PASSWORD') or os.getenv('EMAIL_PASSWORD')
    
    if not email or not password:
        raise ValueError("Gmail credentials not found. Please set GMAIL_EMAIL and GMAIL_APP_PASSWORD environment variables")
    
    return email, password


def get_recipient_list() -> List[str]:
    """Get recipient email list from environment or config"""
    recipients_str = os.getenv('EMAIL_RECIPIENTS', '')
    if not recipients_str:
        # Try alternative environment variable names
        recipients_str = os.getenv('RECIPIENTS', '') or os.getenv('EMAIL_TO', '')
    
    if recipients_str:
        recipients = [email.strip() for email in recipients_str.split(',') if email.strip()]
    else:
        recipients = []
    
    if not recipients:
        logger.warning("No email recipients configured. Using default placeholder.")
        recipients = ['your-email@example.com']
    
    return recipients


def initialize_email_system() -> bool:
    """Initialize the global email system"""
    try:
        return email_manager.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize email system: {e}")
        return False


def send_automated_report(model_reports: Dict[str, Any]) -> bool:
    """Send automated trading report using initialized email system"""
    if not email_manager.is_initialized():
        logger.error("Email manager not initialized")
        return False
    
    return email_manager.send_trading_report(model_reports=model_reports)


def send_comprehensive_report(model_reports: Dict[str, Any], 
                             sender_email: str = None,
                             sender_password: str = None,
                             recipients: List[str] = None) -> bool:
    """Send comprehensive trading report with all available data"""
    
    try:
        # Get credentials if not provided
        if not sender_email or not sender_password:
            sender_email, sender_password = get_gmail_credentials()
        
        # Get recipients if not provided
        if not recipients:
            recipients = get_recipient_list()
        
        # Send the report
        return send_trading_reports_email(
            sender_email, sender_password, recipients, model_reports
        )
        
    except Exception as e:
        logger.error(f"Failed to send comprehensive report: {e}")
        return False


# Backward compatibility aliases
EmailFormatter = ComprehensiveEmailFormatter
EmailSender = EmailManager

# Additional helper functions for the main application
def check_email_configuration() -> bool:
    """Check if email is properly configured"""
    try:
        get_gmail_credentials()
        recipients = get_recipient_list()
        return len(recipients) > 0 and recipients[0] != 'your-email@example.com'
    except Exception:
        return False


def get_email_status() -> Dict[str, Any]:
    """Get detailed email system status"""
    try:
        email, password = get_gmail_credentials()
        recipients = get_recipient_list()
        
        return {
            'configured': True,
            'sender_email': email,
            'recipient_count': len(recipients),
            'recipients': recipients,
            'manager_initialized': email_manager.is_initialized(),
            'test_recipient': recipients[0] if recipients else None
        }
    except Exception as e:
        return {
            'configured': False,
            'error': str(e),
            'manager_initialized': False
        }


def send_startup_notification() -> bool:
    """Send notification that the system has started"""
    try:
        if not email_manager.is_initialized():
            email_manager.initialize()
        
        subject = "Trading Reports System Started"
        html_body = f"""
        <html>
        <body>
        <h2>Trading Reports System</h2>
        <p><strong>Status:</strong> System started successfully</p>
        <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>The automated trading reports system is now running and will send daily reports.</p>
        </body>
        </html>
        """
        
        plain_text = f"Trading Reports System started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return email_manager.send_email(
            email_manager.sender_email, email_manager.sender_password,
            email_manager.recipients, subject, html_body, plain_text
        )
    except Exception as e:
        logger.error(f"Failed to send startup notification: {e}")
        return False


def send_error_notification(error_message: str) -> bool:
    """Send notification about system errors"""
    try:
        if not email_manager.is_initialized():
            email_manager.initialize()
        
        subject = "Trading Reports System - Error Alert"
        html_body = f"""
        <html>
        <body>
        <h2 style="color: red;">Trading Reports System - Error</h2>
        <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Error:</strong></p>
        <pre style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">{error_message}</pre>
        <p>Please check the system logs for more details.</p>
        </body>
        </html>
        """
        
        plain_text = f"Trading Reports System Error at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {error_message}"
        
        return email_manager.send_email(
            email_manager.sender_email, email_manager.sender_password,
            email_manager.recipients, subject, html_body, plain_text
        )
    except Exception as e:
        logger.error(f"Failed to send error notification: {e}")
        return False