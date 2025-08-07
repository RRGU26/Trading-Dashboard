"""
Daily Trading Analysis & Options Strategy Recommendation System
Creates personalized analysis and options strategies after trading models run
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class OptionsStrategy:
    """Options strategy recommendation"""
    strategy_name: str
    action: str
    underlying: str
    current_price: float
    target_price: float
    expiration_days: int
    strategy_type: str
    strikes: List[float]
    max_profit: Optional[float]
    max_loss: Optional[float]
    breakeven: Optional[float]
    probability_profit: Optional[float]
    risk_reward_ratio: Optional[float]
    reasoning: str

@dataclass
class TradingRecommendation:
    """Complete trading recommendation"""
    asset: str
    action: str
    confidence: float
    price_target: float
    stop_loss: float
    time_horizon: str
    rationale: str
    risk_level: str
    position_size: str
    options_strategies: List[OptionsStrategy]

class TradingAnalysisEngine:
    """Analyzes trading model outputs and generates personalized recommendations"""
    
    def __init__(self, reports_db_path: str = None):
        self.reports_db_path = reports_db_path or "reports_tracking.db"
        self.market_data_path = "market_data.db"
        
    def analyze_model_performance(self, model_reports: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model performance and accuracy"""
        performance_analysis = {
            'high_confidence_signals': [],
            'conflicting_signals': [],
            'model_accuracy_scores': {},
            'consensus_strength': 0.0,
            'risk_assessment': 'MEDIUM'
        }
        
        # Extract signals and confidence levels
        signals = {}
        confidences = {}
        
        for model_key, data in model_reports.items():
            if isinstance(data, dict):
                # Extract signal
                signal = data.get('signal', data.get('suggested_action', 'HOLD')).upper()
                signals[model_key] = signal
                
                # Extract confidence
                confidence = self._extract_confidence(data)
                confidences[model_key] = confidence
                
                # Store in performance analysis
                performance_analysis['model_accuracy_scores'][model_key] = confidence
        
        # Identify high confidence signals (>75%)
        for model, conf in confidences.items():
            if conf > 75:
                performance_analysis['high_confidence_signals'].append({
                    'model': model,
                    'signal': signals.get(model, 'HOLD'),
                    'confidence': conf
                })
        
        # Identify conflicting signals
        unique_signals = set(signals.values())
        if len(unique_signals) > 2:  # More than 2 different signals
            performance_analysis['conflicting_signals'] = [
                {'model': model, 'signal': signal} 
                for model, signal in signals.items()
            ]
        
        # Calculate consensus strength
        if signals:
            most_common_signal = max(set(signals.values()), key=list(signals.values()).count)
            consensus_count = list(signals.values()).count(most_common_signal)
            performance_analysis['consensus_strength'] = consensus_count / len(signals)
        
        # Risk assessment
        avg_confidence = np.mean(list(confidences.values())) if confidences else 50
        if avg_confidence > 80 and performance_analysis['consensus_strength'] > 0.7:
            performance_analysis['risk_assessment'] = 'LOW'
        elif avg_confidence < 50 or len(performance_analysis['conflicting_signals']) > 2:
            performance_analysis['risk_assessment'] = 'HIGH'
        
        return performance_analysis
    
    def _extract_confidence(self, data: Dict[str, Any]) -> float:
        """Extract confidence score from model data"""
        # Try multiple confidence field names
        confidence_fields = ['confidence', 'prediction_confidence', 'accuracy', 'hit_rate']
        
        for field in confidence_fields:
            if field in data:
                value = data[field]
                # Handle percentage strings
                if isinstance(value, str) and '%' in value:
                    return float(value.replace('%', ''))
                # Handle decimal values
                elif isinstance(value, (int, float)):
                    if value <= 1.0:
                        return value * 100
                    return value
        
        return 50.0  # Default confidence
    
    def generate_options_strategies(self, recommendation: TradingRecommendation) -> List[OptionsStrategy]:
        """Generate options strategies based on trading recommendation"""
        strategies = []
        
        current_price = recommendation.price_target * 0.98  # Assume slight discount from target
        
        if recommendation.action.upper() == 'BUY':
            strategies.extend(self._generate_bullish_strategies(
                recommendation.asset, current_price, recommendation.price_target,
                recommendation.confidence, recommendation.risk_level
            ))
        elif recommendation.action.upper() == 'SELL':
            strategies.extend(self._generate_bearish_strategies(
                recommendation.asset, current_price, recommendation.price_target,
                recommendation.confidence, recommendation.risk_level
            ))
        else:  # HOLD
            strategies.extend(self._generate_neutral_strategies(
                recommendation.asset, current_price,
                recommendation.confidence, recommendation.risk_level
            ))
        
        return strategies
    
    def _generate_bullish_strategies(self, asset: str, current_price: float, 
                                   target_price: float, confidence: float, 
                                   risk_level: str) -> List[OptionsStrategy]:
        """Generate bullish options strategies"""
        strategies = []
        
        # Calculate days to expiration based on time horizon and risk level
        if risk_level == 'HIGH':
            exp_days = 14  # Short-term for high risk
        elif risk_level == 'LOW':
            exp_days = 45  # Longer term for low risk
        else:
            exp_days = 30  # Medium term
        
        # Long Call Strategy
        otm_strike = current_price * 1.02  # 2% OTM
        strategies.append(OptionsStrategy(
            strategy_name="Long Call",
            action="BUY TO OPEN",
            underlying=asset,
            current_price=current_price,
            target_price=target_price,
            expiration_days=exp_days,
            strategy_type="Bullish",
            strikes=[otm_strike],
            max_profit=None,  # Unlimited
            max_loss=current_price * 0.02,  # Assume 2% premium
            breakeven=otm_strike + (current_price * 0.02),
            probability_profit=confidence / 100 * 0.8,  # Adjust for time decay
            risk_reward_ratio=5.0,
            reasoning=f"Bullish outlook with {confidence}% confidence. Limited risk with unlimited upside potential."
        ))
        
        # Bull Call Spread (if high confidence)
        if confidence > 75:
            long_strike = current_price * 1.01
            short_strike = current_price * 1.05
            strategies.append(OptionsStrategy(
                strategy_name="Bull Call Spread",
                action="BUY/SELL",
                underlying=asset,
                current_price=current_price,
                target_price=target_price,
                expiration_days=exp_days,
                strategy_type="Bullish",
                strikes=[long_strike, short_strike],
                max_profit=(short_strike - long_strike) * 100 - 50,  # Assume $50 net debit
                max_loss=50,  # Net debit
                breakeven=long_strike + 0.5,
                probability_profit=confidence / 100 * 0.9,
                risk_reward_ratio=3.0,
                reasoning=f"High confidence ({confidence}%) bullish play with defined risk/reward."
            ))
        
        return strategies
    
    def _generate_bearish_strategies(self, asset: str, current_price: float, 
                                   target_price: float, confidence: float, 
                                   risk_level: str) -> List[OptionsStrategy]:
        """Generate bearish options strategies"""
        strategies = []
        
        exp_days = 30 if risk_level == 'MEDIUM' else (14 if risk_level == 'HIGH' else 45)
        
        # Long Put Strategy
        otm_put_strike = current_price * 0.98  # 2% OTM
        strategies.append(OptionsStrategy(
            strategy_name="Long Put",
            action="BUY TO OPEN",
            underlying=asset,
            current_price=current_price,
            target_price=target_price,
            expiration_days=exp_days,
            strategy_type="Bearish",
            strikes=[otm_put_strike],
            max_profit=otm_put_strike * 100 - (current_price * 0.02),
            max_loss=current_price * 0.02,
            breakeven=otm_put_strike - (current_price * 0.02),
            probability_profit=confidence / 100 * 0.8,
            risk_reward_ratio=4.0,
            reasoning=f"Bearish outlook with {confidence}% confidence. Limited risk with substantial downside profit potential."
        ))
        
        # Bear Put Spread (if moderate to high confidence)
        if confidence > 60:
            long_strike = current_price * 0.99
            short_strike = current_price * 0.95
            strategies.append(OptionsStrategy(
                strategy_name="Bear Put Spread",
                action="BUY/SELL",
                underlying=asset,
                current_price=current_price,
                target_price=target_price,
                expiration_days=exp_days,
                strategy_type="Bearish",
                strikes=[long_strike, short_strike],
                max_profit=(long_strike - short_strike) * 100 - 40,
                max_loss=40,
                breakeven=long_strike - 0.4,
                probability_profit=confidence / 100 * 0.85,
                risk_reward_ratio=2.5,
                reasoning=f"Moderate to high confidence ({confidence}%) bearish play with defined risk."
            ))
        
        return strategies
    
    def _generate_neutral_strategies(self, asset: str, current_price: float, 
                                   confidence: float, risk_level: str) -> List[OptionsStrategy]:
        """Generate neutral/sideways strategies"""
        strategies = []
        
        exp_days = 30
        
        # Iron Condor for range-bound movement
        strategies.append(OptionsStrategy(
            strategy_name="Iron Condor",
            action="SELL/BUY SPREAD",
            underlying=asset,
            current_price=current_price,
            target_price=current_price,
            expiration_days=exp_days,
            strategy_type="Neutral",
            strikes=[current_price * 0.97, current_price * 0.99, current_price * 1.01, current_price * 1.03],
            max_profit=120,  # Credit received
            max_loss=80,     # Width minus credit
            breakeven=current_price * 0.988,  # Approximate
            probability_profit=0.65,
            risk_reward_ratio=1.5,
            reasoning=f"Neutral outlook expecting {asset} to trade sideways. Profit from time decay and low volatility."
        ))
        
        return strategies
    
    def create_personalized_analysis(self, model_reports: Dict[str, Any]) -> str:
        """Create personalized trading analysis text"""
        
        performance = self.analyze_model_performance(model_reports)
        
        analysis = f"""
🎯 **PERSONALIZED TRADING ANALYSIS** - {datetime.now().strftime('%Y-%m-%d')}

## Executive Summary
Based on today's model outputs, here's your personalized trading strategy:

**Overall Market Sentiment:** {self._get_market_sentiment(model_reports)}
**Risk Level:** {performance['risk_assessment']}
**Model Consensus Strength:** {performance['consensus_strength']:.1%}

## Top Recommendations

"""
        
        # Generate recommendations for each high-confidence signal
        recommendations = self._generate_trading_recommendations(model_reports, performance)
        
        for i, rec in enumerate(recommendations[:3], 1):  # Top 3 recommendations
            analysis += f"""
### {i}. {rec.asset} - {rec.action}
- **Confidence:** {rec.confidence:.0f}%
- **Price Target:** {rec.price_target}
- **Stop Loss:** {rec.stop_loss}
- **Time Horizon:** {rec.time_horizon}
- **Position Size:** {rec.position_size}
- **Rationale:** {rec.rationale}

**Options Strategies:**
"""
            
            # Add options strategies
            options_strategies = self.generate_options_strategies(rec)
            for strategy in options_strategies[:2]:  # Show top 2 strategies
                analysis += f"""
- **{strategy.strategy_name}** ({strategy.strategy_type})
  - Action: {strategy.action}
  - Strikes: {strategy.strikes}
  - Max Profit: ${"${:.0f}".format(strategy.max_profit) if strategy.max_profit else "Unlimited"}
  - Max Loss: ${strategy.max_loss:.0f}
  - Breakeven: ${strategy.breakeven:.2f}
  - Win Probability: {strategy.probability_profit:.1%}
  - Risk/Reward: {strategy.risk_reward_ratio:.1f}:1
  - Reasoning: {strategy.reasoning}
"""
        
        # Add performance insights
        analysis += f"""

## Model Performance Insights

"""
        
        if performance['high_confidence_signals']:
            analysis += "**High Confidence Signals:**\n"
            for signal in performance['high_confidence_signals']:
                analysis += f"- {signal['model']}: {signal['signal']} ({signal['confidence']:.0f}% confidence)\n"
        
        if performance['conflicting_signals']:
            analysis += "\n**⚠️ Conflicting Signals Detected:**\n"
            for signal in performance['conflicting_signals']:
                analysis += f"- {signal['model']}: {signal['signal']}\n"
            analysis += "*Consider waiting for clearer consensus or using smaller position sizes.*\n"
        
        # Add risk management section
        analysis += f"""

## Risk Management Guidelines

Given today's {performance['risk_assessment']} risk environment:

- **Position Sizing:** {"Conservative (1-2% per trade)" if performance['risk_assessment'] == 'HIGH' else "Moderate (2-3% per trade)" if performance['risk_assessment'] == 'MEDIUM' else "Standard (3-5% per trade)"}
- **Stop Losses:** {"Tight (1-2%)" if performance['risk_assessment'] == 'HIGH' else "Standard (2-3%)" if performance['risk_assessment'] == 'MEDIUM' else "Wide (3-5%)"}
- **Diversification:** Spread risk across {2 if performance['risk_assessment'] == 'HIGH' else 3 if performance['risk_assessment'] == 'MEDIUM' else 4} different positions

## Market Context

"""
        
        # Add market context from the models
        context_items = []
        for model_key, data in model_reports.items():
            if isinstance(data, dict):
                # Extract relevant market context
                if 'vix_value' in data:
                    context_items.append(f"VIX: {data['vix_value']} (Volatility)")
                if 'rsi_value' in data:
                    context_items.append(f"RSI: {data['rsi_value']} (Momentum)")
                if 'current_price' in data:
                    context_items.append(f"{model_key.upper()} Price: ${data['current_price']}")
        
        if context_items:
            for item in context_items[:5]:  # Show top 5 context items
                analysis += f"- {item}\n"
        
        analysis += f"""

---
*This analysis is generated based on your trading models' outputs and is for educational purposes only. 
Always conduct your own research and consider your risk tolerance before making trading decisions.*

**Next Analysis:** Tomorrow at 4:00 PM ET after model execution
"""
        
        return analysis
    
    def _get_market_sentiment(self, model_reports: Dict[str, Any]) -> str:
        """Determine overall market sentiment from model reports"""
        signals = []
        
        for model_key, data in model_reports.items():
            if isinstance(data, dict):
                signal = data.get('signal', data.get('suggested_action', 'HOLD')).upper()
                signals.append(signal)
        
        if not signals:
            return "Neutral"
        
        buy_signals = signals.count('BUY')
        sell_signals = signals.count('SELL')
        hold_signals = signals.count('HOLD')
        
        total = len(signals)
        
        if buy_signals / total > 0.6:
            return "Bullish"
        elif sell_signals / total > 0.6:
            return "Bearish"
        elif hold_signals / total > 0.6:
            return "Cautious/Neutral"
        else:
            return "Mixed"
    
    def _generate_trading_recommendations(self, model_reports: Dict[str, Any], 
                                        performance: Dict[str, Any]) -> List[TradingRecommendation]:
        """Generate specific trading recommendations"""
        recommendations = []
        
        # Map of model keys to asset names and current prices
        asset_mapping = {
            'nvidia': ('NVDA', 180.0),
            'longhorn': ('QQQ', 564.1),
            'trading_signal': ('QQQ', 564.1),
            'wishing_wealth': ('QQQ', 564.1),
            'bitcoin': ('BTC', 113585.0),
            'algorand': ('ALGO', 0.35)
        }
        
        for model_key, data in model_reports.items():
            if not isinstance(data, dict) or model_key not in asset_mapping:
                continue
            
            asset_name, current_price = asset_mapping[model_key]
            signal = data.get('signal', data.get('suggested_action', 'HOLD')).upper()
            confidence = self._extract_confidence(data)
            
            # Skip low confidence signals
            if confidence < 35:
                continue
            
            # Determine price target
            price_target = current_price
            if 'predicted_1_day_price' in data:
                price_target = float(str(data['predicted_1_day_price']).replace('$', '').replace(',', ''))
            elif 'target_price' in data:
                price_target = float(str(data['target_price']).replace('$', '').replace(',', ''))
            
            # Calculate stop loss
            if signal == 'BUY':
                stop_loss = current_price * 0.97  # 3% stop loss
            elif signal == 'SELL':
                stop_loss = current_price * 1.03  # 3% stop loss (for short positions)
            else:
                stop_loss = current_price * 0.95  # 5% stop loss for holds
            
            # Determine position size based on confidence and risk
            if confidence > 80 and performance['risk_assessment'] == 'LOW':
                position_size = "Large (4-5%)"
            elif confidence > 65:
                position_size = "Medium (2-3%)"
            else:
                position_size = "Small (1-2%)"
            
            # Generate rationale
            rationale = self._generate_rationale(model_key, data, signal, confidence)
            
            recommendation = TradingRecommendation(
                asset=asset_name,
                action=signal,
                confidence=confidence,
                price_target=price_target,
                stop_loss=stop_loss,
                time_horizon="1-5 days",
                rationale=rationale,
                risk_level=performance['risk_assessment'],
                position_size=position_size,
                options_strategies=[]
            )
            
            recommendations.append(recommendation)
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        return recommendations
    
    def _generate_rationale(self, model_key: str, data: Dict[str, Any], 
                          signal: str, confidence: float) -> str:
        """Generate rationale for recommendation"""
        model_name = {
            'nvidia': 'NVIDIA Bull Momentum Model',
            'longhorn': 'Long Bull Model',
            'trading_signal': 'QQQ Trading Signal',
            'wishing_wealth': 'Wishing Well Model',
            'bitcoin': 'Bitcoin Model',
            'algorand': 'Algorand Model'
        }.get(model_key, model_key)
        
        rationale = f"{model_name} shows {signal} signal with {confidence:.0f}% confidence. "
        
        # Add specific insights based on available data
        insights = []
        if 'predicted_1_day_return' in data:
            return_val = data['predicted_1_day_return']
            insights.append(f"Expected 1-day return: {return_val}")
        
        if 'rsi_value' in data:
            rsi = float(data['rsi_value'])
            if rsi > 70:
                insights.append("RSI indicates overbought conditions")
            elif rsi < 30:
                insights.append("RSI indicates oversold conditions")
        
        if 'vix_value' in data:
            vix = float(data['vix_value'])
            if vix < 20:
                insights.append("Low volatility environment")
            elif vix > 30:
                insights.append("High volatility environment")
        
        if insights:
            rationale += " ".join(insights[:2]) + "."  # Limit to 2 insights
        
        return rationale

class EmailIntegration:
    """Integration with existing email system"""
    
    def __init__(self, email_module_path: str = None):
        # Import the trading reports email module
        try:
            import sys
            if email_module_path:
                sys.path.append(os.path.dirname(email_module_path))
            
            from trading_reports_email import EmailManager, send_trading_reports_email
            self.email_manager = EmailManager()
            self.send_function = send_trading_reports_email
        except ImportError as e:
            logger.error(f"Could not import email module: {e}")
            self.email_manager = None
            self.send_function = None
    
    def send_analysis_email(self, analysis_text: str, model_reports: Dict[str, Any],
                          recipient_email: str = None) -> bool:
        """Send personalized analysis via email"""
        
        if not self.email_manager:
            logger.error("Email manager not available")
            return False
        
        try:
            # Initialize email manager if needed
            if not self.email_manager.is_initialized():
                self.email_manager.initialize()
            
            # Create enhanced model reports with analysis
            enhanced_reports = model_reports.copy()
            enhanced_reports['personal_analysis'] = {
                'report_date': datetime.now().strftime('%Y-%m-%d'),
                'analysis_text': analysis_text,
                'report_type': 'Personal Trading Analysis'
            }
            
            # Send email with enhanced subject
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # If specific recipient provided, use it
            recipients = [recipient_email] if recipient_email else None
            
            return self.email_manager.send_trading_report(
                model_reports=enhanced_reports,
                dashboard_url="https://your-dashboard.com",
                db_status_message="Database updated successfully",
                performance_summary=f"Personal analysis generated for {current_date}"
            )
            
        except Exception as e:
            logger.error(f"Failed to send analysis email: {e}")
            return False

class AutomatedScheduler:
    """Scheduler to run analysis after wrapper execution"""
    
    def __init__(self, wrapper_log_path: str = None):
        self.wrapper_log_path = wrapper_log_path or "wrapper_execution.log"
        self.last_check_time = None
        self.analysis_engine = TradingAnalysisEngine()
        self.email_integration = EmailIntegration()
    
    def check_for_new_reports(self) -> bool:
        """Check if new reports have been generated"""
        # This would monitor the wrapper execution log or database for new entries
        # For now, we'll implement a simple time-based check
        
        current_time = datetime.now()
        
        # If it's after 4 PM and we haven't run today's analysis
        if (current_time.hour >= 16 and 
            (self.last_check_time is None or 
             self.last_check_time.date() < current_time.date())):
            return True
        
        return False
    
    def run_daily_analysis(self, recipient_email: str = None) -> bool:
        """Run the complete daily analysis and send email"""
        
        try:
            # Load latest model reports
            model_reports = self._load_latest_reports()
            
            if not model_reports:
                logger.warning("No model reports found for analysis")
                return False
            
            # Generate personalized analysis
            analysis_text = self.analysis_engine.create_personalized_analysis(model_reports)
            
            # Send email
            success = self.email_integration.send_analysis_email(
                analysis_text, model_reports, recipient_email
            )
            
            if success:
                self.last_check_time = datetime.now()
                logger.info("Daily analysis completed and sent successfully")
            else:
                logger.error("Failed to send daily analysis email")
            
            return success
            
        except Exception as e:
            logger.error(f"Error running daily analysis: {e}")
            return False
    
    def _load_latest_reports(self) -> Dict[str, Any]:
        """Load the latest model reports from database or files"""
        
        # Try to load from database first
        try:
            if os.path.exists(self.analysis_engine.reports_db_path):
                return self._load_from_database()
        except Exception as e:
            logger.warning(f"Could not load from database: {e}")
        
        # Fallback to loading from log files or direct model outputs
        return self._load_from_files()
    
    def _load_from_database(self) -> Dict[str, Any]:
        """Load reports from the tracking database"""
        reports = {}
        
        try:
            conn = sqlite3.connect(self.analysis_engine.reports_db_path)
            cursor = conn.cursor()
            
            # Get latest reports for each model
            query = """
            SELECT model_name, report_data, report_date 
            FROM reports 
            WHERE report_date = (
                SELECT MAX(report_date) 
                FROM reports r2 
                WHERE r2.model_name = reports.model_name
            )
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            for model_name, report_data, report_date in rows:
                try:
                    data = json.loads(report_data)
                    data['report_date'] = report_date
                    reports[model_name.lower()] = data
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse JSON for {model_name}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Database error: {e}")
        
        return reports
    
    def _load_from_files(self) -> Dict[str, Any]:
        """Load reports from text files in user directory"""
        reports = {}
        
        # Define file patterns for each model
        file_patterns = {
            'nvidia': ['*nvidia*', '*NVDA*'],
            'longhorn': ['*longhorn*', '*long_bull*'],
            'trading_signal': ['*qqq*signal*', '*trading*signal*'],
            'wishing_wealth': ['*wishing*well*', '*wishing*wealth*'],
            'bitcoin': ['*bitcoin*', '*btc*'],
            'algorand': ['*algorand*', '*algo*']
        }
        
        # Search for report files
        base_path = os.path.expanduser("~")
        
        for model_key, patterns in file_patterns.items():
            for pattern in patterns:
                files = self._find_files(base_path, pattern)
                if files:
                    # Get the most recent file
                    latest_file = max(files, key=os.path.getmtime)
                    data = self._parse_report_file(latest_file)
                    if data:
                        reports[model_key] = data
                    break
        
        return reports
    
    def _find_files(self, base_path: str, pattern: str) -> List[str]:
        """Find files matching pattern"""
        import glob
        return glob.glob(os.path.join(base_path, f"**/{pattern}.txt"), recursive=True)
    
    def _parse_report_file(self, file_path: str) -> Dict[str, Any]:
        """Parse a report file and extract key metrics"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract key-value pairs from the content
            data = {}
            lines = content.split('\n')
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_').replace('•', '')
                    value = value.strip()
                    
                    if value and value != "N/A":
                        data[key] = value
            
            # Add file metadata
            data['file_path'] = file_path
            data['report_date'] = datetime.now().strftime('%Y-%m-%d')
            
            return data
            
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return {}

def main():
    """Main function to run daily analysis"""
    
    # Configure recipient email (you can set this as environment variable)
    recipient_email = os.getenv('TRADING_ANALYSIS_EMAIL', 'rrose@example.com')
    
    # Create scheduler and run analysis
    scheduler = AutomatedScheduler()
    
    # For testing, force run the analysis
    success = scheduler.run_daily_analysis(recipient_email)
    
    if success:
        print("✅ Daily trading analysis completed and sent successfully!")
    else:
        print("❌ Failed to complete daily trading analysis")

if __name__ == "__main__":
    main()