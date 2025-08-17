#!/usr/bin/env python3
"""
Options Trading Strategy Analyzer
Analyzes daily trading reports and provides specific options strategies
Based on model predictions, VIX levels, and market conditions
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import sqlite3
import logging
from dataclasses import dataclass
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptionsStrategy:
    """Represents a specific options strategy recommendation"""
    name: str
    description: str
    entry_price: float
    max_profit: float
    max_loss: float
    breakeven: List[float]
    probability_of_profit: float
    risk_level: str
    time_decay_impact: str
    volatility_impact: str
    underlying_direction: str
    expiration_recommended: str
    strike_selection: str
    position_size: str
    exit_criteria: str
    market_conditions: str

class OptionsStrategyAnalyzer:
    """Analyzes trading reports and generates options strategy recommendations"""
    
    def __init__(self, db_path: str = None):
        self.db_path = self._find_database() if not db_path else db_path
        self.current_vix = self._get_current_vix()
        self.qqq_price = self._get_current_qqq_price()
        
    def _find_database(self) -> str:
        """Find the trading database"""
        possible_paths = [
            os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "models_dashboard.db"),
            os.path.join(os.path.expanduser("~"), "Desktop", "models_dashboard.db"),
            os.path.join(os.path.dirname(__file__), "models_dashboard.db")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def _get_current_vix(self) -> float:
        """Get current VIX level from database or data_fetcher"""
        
        # Try data_fetcher first for most current VIX
        try:
            import data_fetcher
            vix_price = data_fetcher.get_current_price('^VIX')
            if vix_price and vix_price > 0:
                return float(vix_price)
        except:
            pass
            
        # Fallback to database
        if not self.db_path:
            return 15.15  # Default to recent known VIX
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT close_price 
                FROM daily_prices 
                WHERE symbol IN ('^VIX', 'VIX') 
                ORDER BY date DESC 
                LIMIT 1
            """)
            
            result = cursor.fetchone()
            conn.close()
            
            return float(result[0]) if result else 15.15
        except:
            return 15.15
    
    def _get_current_qqq_price(self) -> float:
        """Get current QQQ price from database or data_fetcher"""
        
        # Try data_fetcher first for most current price
        try:
            import data_fetcher
            current_price = data_fetcher.get_current_price('QQQ')
            if current_price and current_price > 0:
                return float(current_price)
        except:
            pass
            
        # Fallback to database
        if not self.db_path:
            return 574.55  # Default to recent known price
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT close_price 
                FROM daily_prices 
                WHERE symbol = 'QQQ' 
                ORDER BY date DESC 
                LIMIT 1
            """)
            
            result = cursor.fetchone()
            conn.close()
            
            return float(result[0]) if result else 574.55
        except:
            return 574.55
    
    def analyze_model_consensus(self, model_reports: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consensus from all trading models"""
        
        signals = []
        confidence_scores = []
        price_targets = []
        expected_returns = []
        
        model_analysis = {
            'bullish_models': [],
            'bearish_models': [],
            'neutral_models': [],
            'high_confidence_models': [],
            'consensus_direction': 'NEUTRAL',
            'consensus_strength': 0,
            'avg_expected_return': 0,
            'price_target_range': (0, 0),
            'conflicting_signals': False
        }
        
        for model_key, data in model_reports.items():
            if not data or not isinstance(data, dict):
                continue
                
            # Extract signals
            signal = str(data.get('signal', data.get('suggested_action', 'HOLD'))).upper()
            confidence = self._extract_numeric_value(data.get('confidence', '50'))
            expected_return = self._extract_numeric_value(data.get('expected_return', '0'))
            price_target = self._extract_numeric_value(data.get('predicted_price', data.get('target_price', '0')))
            
            if signal in ['BUY', 'STRONG BUY', 'BULLISH']:
                model_analysis['bullish_models'].append(model_key)
            elif signal in ['SELL', 'STRONG SELL', 'BEARISH']:
                model_analysis['bearish_models'].append(model_key)
            else:
                model_analysis['neutral_models'].append(model_key)
                
            if confidence > 70:
                model_analysis['high_confidence_models'].append(model_key)
                
            signals.append(signal)
            confidence_scores.append(confidence)
            expected_returns.append(expected_return)
            if price_target > 0:
                price_targets.append(price_target)
        
        # Calculate consensus
        bullish_count = len(model_analysis['bullish_models'])
        bearish_count = len(model_analysis['bearish_models'])
        total_models = len(signals)
        
        if total_models > 0:
            if bullish_count > bearish_count and bullish_count > total_models * 0.5:
                model_analysis['consensus_direction'] = 'BULLISH'
                model_analysis['consensus_strength'] = bullish_count / total_models
            elif bearish_count > bullish_count and bearish_count > total_models * 0.5:
                model_analysis['consensus_direction'] = 'BEARISH'
                model_analysis['consensus_strength'] = bearish_count / total_models
            else:
                model_analysis['consensus_direction'] = 'MIXED'
                model_analysis['consensus_strength'] = 0.5
                
            model_analysis['avg_expected_return'] = np.mean(expected_returns) if expected_returns else 0
            model_analysis['conflicting_signals'] = (bullish_count > 0 and bearish_count > 0)
            
            if price_targets:
                model_analysis['price_target_range'] = (min(price_targets), max(price_targets))
        
        return model_analysis
    
    def _extract_numeric_value(self, value: Any) -> float:
        """Extract numeric value from various formats"""
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Remove common text and extract numbers
            clean_value = re.sub(r'[^\d.-]', '', value.replace('%', '').replace('$', ''))
            try:
                return float(clean_value) if clean_value else 0.0
            except ValueError:
                return 0.0
        
        return 0.0
    
    def generate_options_strategies(self, model_reports: Dict[str, Any]) -> List[OptionsStrategy]:
        """Generate specific options strategies based on model analysis"""
        
        consensus = self.analyze_model_consensus(model_reports)
        strategies = []
        
        # Market condition factors
        vix_level = "LOW" if self.current_vix < 15 else "HIGH" if self.current_vix > 25 else "NORMAL"
        avg_return = consensus['avg_expected_return']
        direction = consensus['consensus_direction']
        strength = consensus['consensus_strength']
        
        logger.info(f"Analyzing for QQQ=${self.qqq_price:.2f}, VIX={self.current_vix:.1f} ({vix_level})")
        logger.info(f"Consensus: {direction} ({strength:.1%}), Expected Return: {avg_return:+.1f}%")
        
        # Strategy 1: Directional plays based on consensus
        if direction == 'BULLISH' and strength > 0.6:
            if avg_return > 3:  # Strong bullish
                strategies.append(self._create_long_call_strategy())
                if vix_level == "HIGH":
                    strategies.append(self._create_bull_call_spread())
            elif avg_return > 1:  # Moderate bullish
                strategies.append(self._create_bull_call_spread())
                strategies.append(self._create_cash_secured_put())
                
        elif direction == 'BEARISH' and strength > 0.6:
            if avg_return < -3:  # Strong bearish
                strategies.append(self._create_long_put_strategy())
                if vix_level == "HIGH":
                    strategies.append(self._create_bear_put_spread())
            elif avg_return < -1:  # Moderate bearish
                strategies.append(self._create_bear_put_spread())
                strategies.append(self._create_covered_call())
        
        # Strategy 2: VIX-based volatility plays
        if vix_level == "HIGH" and consensus['conflicting_signals']:
            strategies.append(self._create_iron_condor())
            strategies.append(self._create_short_straddle())
        elif vix_level == "LOW" and abs(avg_return) > 2:
            strategies.append(self._create_long_straddle())
            strategies.append(self._create_calendar_spread())
        
        # Strategy 3: Conservative income strategies
        if direction == 'NEUTRAL' or strength < 0.6:
            strategies.append(self._create_covered_call())
            strategies.append(self._create_cash_secured_put())
            if vix_level == "HIGH":
                strategies.append(self._create_iron_condor())
        
        # Strategy 4: Earnings/Event plays (if high expected volatility)
        if vix_level == "HIGH" and abs(avg_return) > 4:
            strategies.append(self._create_butterfly_spread())
            strategies.append(self._create_iron_butterfly())
        
        return strategies[:5]  # Return top 5 strategies
    
    def _create_long_call_strategy(self) -> OptionsStrategy:
        """Create long call strategy"""
        strike = self.qqq_price + (self.qqq_price * 0.02)  # 2% OTM
        premium = self.qqq_price * 0.015  # Estimated premium
        
        return OptionsStrategy(
            name="Long Call",
            description=f"Buy QQQ ${strike:.0f} call options for bullish directional play",
            entry_price=premium,
            max_profit=float('inf'),
            max_loss=premium,
            breakeven=[strike + premium],
            probability_of_profit=0.35,
            risk_level="HIGH",
            time_decay_impact="NEGATIVE",
            volatility_impact="POSITIVE", 
            underlying_direction="BULLISH",
            expiration_recommended="30-45 days",
            strike_selection=f"${strike:.0f} (2% OTM)",
            position_size="2-5% of portfolio",
            exit_criteria="Take profit at 50-100% gain, stop loss at 50% of premium",
            market_conditions=f"VIX: {self.current_vix:.1f}, Strong bullish consensus"
        )
    
    def _create_bull_call_spread(self) -> OptionsStrategy:
        """Create bull call spread strategy"""
        long_strike = self.qqq_price + (self.qqq_price * 0.01)  # 1% OTM
        short_strike = self.qqq_price + (self.qqq_price * 0.04)  # 4% OTM
        net_debit = self.qqq_price * 0.008  # Estimated net debit
        max_profit = (short_strike - long_strike) - net_debit
        
        return OptionsStrategy(
            name="Bull Call Spread",
            description=f"Buy ${long_strike:.0f} call, sell ${short_strike:.0f} call for limited risk bullish play",
            entry_price=net_debit,
            max_profit=max_profit,
            max_loss=net_debit,
            breakeven=[long_strike + net_debit],
            probability_of_profit=0.45,
            risk_level="MEDIUM",
            time_decay_impact="NEUTRAL",
            volatility_impact="NEUTRAL",
            underlying_direction="MODERATELY BULLISH",
            expiration_recommended="30-45 days",
            strike_selection=f"Long ${long_strike:.0f}, Short ${short_strike:.0f}",
            position_size="3-7% of portfolio",
            exit_criteria="Take profit at 50% of max profit, manage at 21 DTE",
            market_conditions=f"VIX: {self.current_vix:.1f}, Moderate bullish bias"
        )
    
    def _create_long_put_strategy(self) -> OptionsStrategy:
        """Create long put strategy"""
        strike = self.qqq_price - (self.qqq_price * 0.02)  # 2% OTM
        premium = self.qqq_price * 0.018  # Estimated premium
        
        return OptionsStrategy(
            name="Long Put",
            description=f"Buy QQQ ${strike:.0f} put options for bearish directional play",
            entry_price=premium,
            max_profit=strike - premium,
            max_loss=premium,
            breakeven=[strike - premium],
            probability_of_profit=0.35,
            risk_level="HIGH",
            time_decay_impact="NEGATIVE",
            volatility_impact="POSITIVE",
            underlying_direction="BEARISH",
            expiration_recommended="30-45 days",
            strike_selection=f"${strike:.0f} (2% OTM)",
            position_size="2-5% of portfolio",
            exit_criteria="Take profit at 50-100% gain, stop loss at 50% of premium",
            market_conditions=f"VIX: {self.current_vix:.1f}, Strong bearish consensus"
        )
    
    def _create_bear_put_spread(self) -> OptionsStrategy:
        """Create bear put spread strategy"""
        long_strike = self.qqq_price - (self.qqq_price * 0.01)  # 1% OTM
        short_strike = self.qqq_price - (self.qqq_price * 0.04)  # 4% OTM
        net_debit = self.qqq_price * 0.008
        max_profit = (long_strike - short_strike) - net_debit
        
        return OptionsStrategy(
            name="Bear Put Spread", 
            description=f"Buy ${long_strike:.0f} put, sell ${short_strike:.0f} put for limited risk bearish play",
            entry_price=net_debit,
            max_profit=max_profit,
            max_loss=net_debit,
            breakeven=[long_strike - net_debit],
            probability_of_profit=0.45,
            risk_level="MEDIUM",
            time_decay_impact="NEUTRAL",
            volatility_impact="NEUTRAL",
            underlying_direction="MODERATELY BEARISH",
            expiration_recommended="30-45 days",
            strike_selection=f"Long ${long_strike:.0f}, Short ${short_strike:.0f}",
            position_size="3-7% of portfolio",
            exit_criteria="Take profit at 50% of max profit, manage at 21 DTE",
            market_conditions=f"VIX: {self.current_vix:.1f}, Moderate bearish bias"
        )
    
    def _create_iron_condor(self) -> OptionsStrategy:
        """Create iron condor strategy"""
        call_short = self.qqq_price + (self.qqq_price * 0.03)
        call_long = self.qqq_price + (self.qqq_price * 0.05) 
        put_short = self.qqq_price - (self.qqq_price * 0.03)
        put_long = self.qqq_price - (self.qqq_price * 0.05)
        net_credit = self.qqq_price * 0.004
        max_loss = ((call_long - call_short) * 100) - (net_credit * 100)
        
        return OptionsStrategy(
            name="Iron Condor",
            description=f"Range-bound strategy: profit if QQQ stays between ${put_short:.0f}-${call_short:.0f}",
            entry_price=-net_credit,  # Credit received
            max_profit=net_credit * 100,
            max_loss=max_loss,
            breakeven=[put_short - net_credit, call_short + net_credit],
            probability_of_profit=0.65,
            risk_level="MEDIUM-LOW",
            time_decay_impact="POSITIVE",
            volatility_impact="NEGATIVE",
            underlying_direction="NEUTRAL/RANGE-BOUND",
            expiration_recommended="30-45 days",
            strike_selection=f"Short ${put_short:.0f}P/${call_short:.0f}C, Long ${put_long:.0f}P/${call_long:.0f}C",
            position_size="5-10% of portfolio",
            exit_criteria="Take profit at 25-50% of credit, close at 21 DTE",
            market_conditions=f"VIX: {self.current_vix:.1f}, High volatility, expect consolidation"
        )
    
    def _create_covered_call(self) -> OptionsStrategy:
        """Create covered call strategy"""
        strike = self.qqq_price + (self.qqq_price * 0.02)  # 2% OTM
        premium = self.qqq_price * 0.008  # Estimated premium
        
        return OptionsStrategy(
            name="Covered Call",
            description=f"Own QQQ shares, sell ${strike:.0f} calls for income generation",
            entry_price=-premium,  # Credit received
            max_profit=(strike - self.qqq_price) + premium,
            max_loss=self.qqq_price - premium,  # If QQQ goes to zero
            breakeven=[self.qqq_price - premium],
            probability_of_profit=0.70,
            risk_level="LOW-MEDIUM",
            time_decay_impact="POSITIVE",
            volatility_impact="POSITIVE",
            underlying_direction="NEUTRAL TO SLIGHTLY BULLISH",
            expiration_recommended="30-45 days",
            strike_selection=f"${strike:.0f} (2% OTM)",
            position_size="Against existing QQQ position",
            exit_criteria="Buy back at 50% profit or let expire if OTM",
            market_conditions=f"VIX: {self.current_vix:.1f}, Generate income on QQQ position"
        )
    
    def _create_cash_secured_put(self) -> OptionsStrategy:
        """Create cash secured put strategy"""
        strike = self.qqq_price - (self.qqq_price * 0.02)  # 2% OTM
        premium = self.qqq_price * 0.008
        
        return OptionsStrategy(
            name="Cash Secured Put",
            description=f"Sell ${strike:.0f} puts to potentially acquire QQQ at discount",
            entry_price=-premium,  # Credit received
            max_profit=premium,
            max_loss=strike - premium,
            breakeven=[strike - premium],
            probability_of_profit=0.70,
            risk_level="LOW-MEDIUM", 
            time_decay_impact="POSITIVE",
            volatility_impact="POSITIVE",
            underlying_direction="NEUTRAL TO SLIGHTLY BULLISH",
            expiration_recommended="30-45 days",
            strike_selection=f"${strike:.0f} (2% OTM)",
            position_size="Secure ${strike*100:.0f} cash per contract",
            exit_criteria="Buy back at 50% profit or let expire if OTM",
            market_conditions=f"VIX: {self.current_vix:.1f}, Willing to own QQQ at discount"
        )
    
    def _create_long_straddle(self) -> OptionsStrategy:
        """Create long straddle strategy"""
        strike = self.qqq_price  # ATM
        premium = self.qqq_price * 0.025  # Estimated total premium
        
        return OptionsStrategy(
            name="Long Straddle",
            description=f"Buy ${strike:.0f} call and put for big move in either direction",
            entry_price=premium,
            max_profit=float('inf'),
            max_loss=premium,
            breakeven=[strike - premium, strike + premium],
            probability_of_profit=0.30,
            risk_level="HIGH",
            time_decay_impact="NEGATIVE",
            volatility_impact="POSITIVE",
            underlying_direction="BIG MOVE (EITHER DIRECTION)",
            expiration_recommended="30-45 days or before earnings",
            strike_selection=f"${strike:.0f} ATM",
            position_size="2-4% of portfolio",
            exit_criteria="Take profit at 50-100% gain, manage before earnings",
            market_conditions=f"VIX: {self.current_vix:.1f}, Expecting large move"
        )
    
    def _create_calendar_spread(self) -> OptionsStrategy:
        """Create calendar spread strategy"""
        strike = self.qqq_price  # ATM
        net_debit = self.qqq_price * 0.005
        
        return OptionsStrategy(
            name="Calendar Spread",
            description=f"Sell front month ${strike:.0f} call, buy back month for time decay profit",
            entry_price=net_debit,
            max_profit=net_debit * 3,  # Estimated
            max_loss=net_debit,
            breakeven=[strike],
            probability_of_profit=0.55,
            risk_level="MEDIUM",
            time_decay_impact="POSITIVE",
            volatility_impact="NEUTRAL",
            underlying_direction="NEUTRAL (EXPECT LOW MOVEMENT)",
            expiration_recommended="Front: 15-30 days, Back: 45-60 days",
            strike_selection=f"${strike:.0f} ATM",
            position_size="3-6% of portfolio",
            exit_criteria="Close before front month expiration, profit from time decay",
            market_conditions=f"VIX: {self.current_vix:.1f}, Low volatility environment"
        )
    
    def _create_butterfly_spread(self) -> OptionsStrategy:
        """Create butterfly spread strategy"""
        center_strike = self.qqq_price
        wing_width = self.qqq_price * 0.02
        net_debit = self.qqq_price * 0.003
        
        return OptionsStrategy(
            name="Butterfly Spread",
            description=f"Profit if QQQ stays near ${center_strike:.0f} at expiration",
            entry_price=net_debit,
            max_profit=wing_width - net_debit,
            max_loss=net_debit,
            breakeven=[center_strike - (wing_width - net_debit), center_strike + (wing_width - net_debit)],
            probability_of_profit=0.40,
            risk_level="MEDIUM",
            time_decay_impact="POSITIVE",
            volatility_impact="NEGATIVE",
            underlying_direction="MINIMAL MOVEMENT",
            expiration_recommended="30-45 days",
            strike_selection=f"${center_strike-wing_width:.0f}/${center_strike:.0f}/${center_strike+wing_width:.0f}",
            position_size="4-8% of portfolio",
            exit_criteria="Take profit at 50% of max profit, close at 21 DTE",
            market_conditions=f"VIX: {self.current_vix:.1f}, Expect consolidation around current price"
        )
    
    def _create_short_straddle(self) -> OptionsStrategy:
        """Create short straddle strategy"""
        strike = self.qqq_price
        premium = self.qqq_price * 0.025
        
        return OptionsStrategy(
            name="Short Straddle",
            description=f"Sell ${strike:.0f} call and put for premium collection (HIGH RISK)",
            entry_price=-premium,  # Credit received
            max_profit=premium,
            max_loss=float('inf'),
            breakeven=[strike - premium, strike + premium],
            probability_of_profit=0.70,
            risk_level="VERY HIGH",
            time_decay_impact="POSITIVE",
            volatility_impact="NEGATIVE",
            underlying_direction="MINIMAL MOVEMENT",
            expiration_recommended="30-45 days",
            strike_selection=f"${strike:.0f} ATM",
            position_size="1-2% of portfolio (HIGH MARGIN REQUIREMENT)",
            exit_criteria="Buy back at 25% profit, manage closely",
            market_conditions=f"VIX: {self.current_vix:.1f}, HIGH volatility premium, expect IV crush"
        )
    
    def _create_iron_butterfly(self) -> OptionsStrategy:
        """Create iron butterfly strategy"""
        center_strike = self.qqq_price
        wing_width = self.qqq_price * 0.025
        net_credit = self.qqq_price * 0.008
        max_loss = (wing_width * 100) - (net_credit * 100)
        
        return OptionsStrategy(
            name="Iron Butterfly",
            description=f"Short straddle with protective wings - profit if QQQ stays at ${center_strike:.0f}",
            entry_price=-net_credit,  # Credit received
            max_profit=net_credit * 100,
            max_loss=max_loss,
            breakeven=[center_strike - net_credit, center_strike + net_credit],
            probability_of_profit=0.50,
            risk_level="MEDIUM-HIGH",
            time_decay_impact="POSITIVE",
            volatility_impact="NEGATIVE",
            underlying_direction="MINIMAL MOVEMENT",
            expiration_recommended="30-45 days",
            strike_selection=f"Short ${center_strike:.0f} straddle, Long ${center_strike-wing_width:.0f}P/${center_strike+wing_width:.0f}C",
            position_size="4-8% of portfolio", 
            exit_criteria="Take profit at 25-50% of credit, close at 21 DTE",
            market_conditions=f"VIX: {self.current_vix:.1f}, Volatility crush expected"
        )
    
    def generate_strategy_report(self, model_reports: Dict[str, Any]) -> str:
        """Generate comprehensive options strategy report"""
        
        strategies = self.generate_options_strategies(model_reports)
        consensus = self.analyze_model_consensus(model_reports)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = f"""
OPTIONS STRATEGY ANALYSIS REPORT
Generated: {timestamp}

MARKET CONDITIONS ANALYSIS
=====================================
QQQ Current Price: ${self.qqq_price:.2f}
VIX Level: {self.current_vix:.1f} ({'LOW' if self.current_vix < 15 else 'HIGH' if self.current_vix > 25 else 'NORMAL'})
Market Regime: {'High Volatility' if self.current_vix > 25 else 'Low Volatility' if self.current_vix < 15 else 'Normal Volatility'}

MODEL CONSENSUS ANALYSIS
=====================================
Direction: {consensus['consensus_direction']}
Strength: {consensus['consensus_strength']:.1%}
Average Expected Return: {consensus['avg_expected_return']:+.1f}%
Conflicting Signals: {'YES' if consensus['conflicting_signals'] else 'NO'}

Bullish Models: {', '.join(consensus['bullish_models']) if consensus['bullish_models'] else 'None'}
Bearish Models: {', '.join(consensus['bearish_models']) if consensus['bearish_models'] else 'None'}  
High Confidence Models: {', '.join(consensus['high_confidence_models']) if consensus['high_confidence_models'] else 'None'}

RECOMMENDED OPTIONS STRATEGIES
====================================="""

        for i, strategy in enumerate(strategies, 1):
            report += f"""

STRATEGY {i}: {strategy.name}
-------------------------------------
Description: {strategy.description}
Risk Level: {strategy.risk_level}
Direction Bias: {strategy.underlying_direction}

Entry Details:
• Entry Cost: ${abs(strategy.entry_price):.2f} {'(Credit Received)' if strategy.entry_price < 0 else '(Debit Paid)'}
• Strike Selection: {strategy.strike_selection}
• Expiration: {strategy.expiration_recommended}
• Position Size: {strategy.position_size}

Profit/Loss Profile:
• Max Profit: ${'Unlimited' if strategy.max_profit == float('inf') else f'{strategy.max_profit:.2f}'}
• Max Loss: ${'Unlimited' if strategy.max_loss == float('inf') else f'{strategy.max_loss:.2f}'}
• Breakeven: {', '.join([f'${be:.2f}' for be in strategy.breakeven])}
• Probability of Profit: {strategy.probability_of_profit:.0%}

Greeks Impact:
• Time Decay: {strategy.time_decay_impact}
• Volatility: {strategy.volatility_impact}

Risk Management:
• Exit Criteria: {strategy.exit_criteria}
• Market Conditions: {strategy.market_conditions}
"""

        report += f"""

ADDITIONAL CONSIDERATIONS
=====================================
1. IMPLIED VOLATILITY: Current VIX of {self.current_vix:.1f} suggests {'high' if self.current_vix > 25 else 'low' if self.current_vix < 15 else 'normal'} options premiums
2. TIME DECAY: With {'high' if self.current_vix > 25 else 'normal'} volatility, consider time decay in strategy selection
3. EARNINGS: Check QQQ earnings calendar for major holdings (AAPL, MSFT, GOOGL, etc.)
4. FEDERAL RESERVE: Monitor Fed meeting dates for potential volatility spikes
5. MARKET CORRELATION: QQQ tracks tech/growth sector - consider sector rotation risks

RISK WARNINGS
=====================================
• Options trading involves significant risk and is not suitable for all investors
• Past performance does not guarantee future results  
• Consider paper trading strategies before risking real capital
• Never risk more than you can afford to lose
• Consult with a financial advisor before making investment decisions

Report ID: OptionsAnalysis_{report_date}
Analyzer Version: 1.0
"""
        
        return report
    
    def save_strategy_report(self, model_reports: Dict[str, Any], output_dir: str = None) -> str:
        """Save the options strategy report to file"""
        
        if not output_dir:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, "reports")
        
        os.makedirs(output_dir, exist_ok=True)
        
        report_content = self.generate_strategy_report(model_reports)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Options_Strategy_Analysis_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Options strategy report saved: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save options strategy report: {e}")
            return None

def main():
    """Main function for standalone testing - PRODUCTION MODE"""
    
    # PRODUCTION MODE - All recipients will receive emails
    # import os
    # os.environ['EMAIL_RECIPIENTS'] = 'RRGU26@gmail.com'  # TEST MODE DISABLED
    print("[PRODUCTION] Production mode: All recipients will receive emails")
    
    analyzer = OptionsStrategyAnalyzer()
    
    # Mock model reports for testing
    mock_reports = {
        'longhorn': {
            'signal': 'STRONG SELL',
            'expected_return': '-4.35',
            'confidence': '65.4'
        },
        'nvidia': {
            'signal': 'BUY', 
            'expected_return': '2.1',
            'confidence': '85'
        },
        'bitcoin': {
            'signal': 'HOLD',
            'expected_return': '0.1', 
            'confidence': '70'
        }
    }
    
    strategies = analyzer.generate_options_strategies(mock_reports)
    
    print("Generated Options Strategies:")
    print("=" * 50)
    
    for i, strategy in enumerate(strategies, 1):
        print(f"\n{i}. {strategy.name}")
        print(f"   Direction: {strategy.underlying_direction}")
        print(f"   Risk Level: {strategy.risk_level}")
        print(f"   Description: {strategy.description}")
    
    # Save full report
    report_path = analyzer.save_strategy_report(mock_reports)
    if report_path:
        print(f"\nFull report saved to: {report_path}")

if __name__ == "__main__":
    main()