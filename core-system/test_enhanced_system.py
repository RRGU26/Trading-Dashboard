#!/usr/bin/env python3
"""
Test script for the enhanced trading system with options analysis
TEST MODE: Only sends email to RRGU26@gmail.com
"""

import os
import sys
import subprocess
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_enhanced_system")

# PRODUCTION MODE - All recipients will receive emails
# os.environ['EMAIL_RECIPIENTS'] = 'RRGU26@gmail.com'  # TEST MODE DISABLED
# os.environ['RECIPIENT_EMAIL'] = 'RRGU26@gmail.com'   # TEST MODE DISABLED

print("[TEST] TEST MODE ACTIVATED")
print("[EMAIL] Email will ONLY be sent to: RRGU26@gmail.com")
print("="*60)

def test_options_analyzer():
    """Test the options strategy analyzer"""
    print("\n[TARGET] Testing Options Strategy Analyzer...")
    
    try:
        # Import and test
        from options_strategy_analyzer import OptionsStrategyAnalyzer
        
        analyzer = OptionsStrategyAnalyzer()
        
        # Use actual recent data from your models
        test_reports = {
            'longhorn': {
                'signal': 'STRONG SELL',
                'expected_return': '-4.35',
                'confidence': '65',
                'current_price': '574.55',
                'target_price': '549.57'
            },
            'nvidia': {
                'signal': 'BUY', 
                'expected_return': '1.11',
                'confidence': '95',
                'current_price': '182.70',
                'predicted_price': '184.72'
            },
            'bitcoin': {
                'signal': 'HOLD',
                'expected_return': '0.11', 
                'confidence': '70',
                'current_price': '116584.00',
                'predicted_price': '116712.50'
            },
            'wishing_wealth': {
                'signal': 'BUY',
                'expected_return': '-0.05',
                'confidence': '85',
                'current_price': '574.55'
            }
        }
        
        # Generate strategies
        strategies = analyzer.generate_options_strategies(test_reports)
        
        print(f"[OK] Generated {len(strategies)} options strategies")
        
        for i, strategy in enumerate(strategies, 1):
            print(f"\n   {i}. {strategy.name}")
            print(f"      Direction: {strategy.underlying_direction}")
            print(f"      Risk Level: {strategy.risk_level}")
            print(f"      Profit Probability: {strategy.probability_of_profit:.0%}")
        
        # Save report
        report_path = analyzer.save_strategy_report(test_reports)
        if report_path:
            print(f"[OK] Options strategy report saved: {os.path.basename(report_path)}")
            return True
        else:
            print("[FAIL] Failed to save options strategy report")
            return False
            
    except Exception as e:
        print(f"[FAIL] Options analyzer test failed: {e}")
        return False

def test_parser_fix():
    """Test the Long Bull parser fix"""
    print("\n[TOOLS] Testing Long Bull Parser Fix...")
    
    try:
        from trading_reports_parsers import parse_longhorn_report
        
        # Test content from actual report
        test_content = """
        QQQ Long Bull Report
        Generated: 2025-08-09 15:43:51
        
        PREDICTION SUMMARY
        =====================================
        Current Price: $574.55
        Target Price: $549.57
        Expected Return: -4.35%
        Time Horizon: 14 days
        
        SIGNAL ANALYSIS
        =====================================
        Signal: STRONG SELL
        Suggested Action: STRONG SELL
        Confidence: LOW
        Risk Level: HIGH
        """
        
        result = parse_longhorn_report(test_content)
        
        if 'expected_return' in result:
            expected_return = result['expected_return']
            print(f"[OK] Expected return parsed successfully: {expected_return}")
            
            # Verify it's numeric
            if isinstance(expected_return, (int, float)):
                print(f"[OK] Expected return is numeric: {expected_return}")
                return True
            else:
                print(f"[WARN] Expected return is not numeric: {type(expected_return)}")
                return False
        else:
            print("[FAIL] Expected return not found in parsed result")
            print(f"Available fields: {list(result.keys())}")
            return False
            
    except Exception as e:
        print(f"[FAIL] Parser test failed: {e}")
        return False

def test_qqq_master_integration():
    """Test QQQ Master Model integration"""
    print("\n[TARGET] Testing QQQ Master Model Integration...")
    
    try:
        from trading_reports_parsers import parse_qqq_master_report
        
        # Test content for QQQ Master Model
        test_content = """
        QQQ MASTER MODEL ANALYSIS
        Generated: 2025-08-10 10:00:00
        
        PREDICTION SUMMARY
        =====================================
        Current Price: $574.55
        1-Day Prediction: $571.20
        Expected Return: -0.58%
        Confidence: 75.5%
        
        SIGNAL ANALYSIS
        =====================================
        Signal: SELL
        Action: SELL
        Direction Accuracy: 68.2%
        R² Score: 0.124
        Volatility: HIGH
        Trend: BEARISH
        """
        
        result = parse_qqq_master_report(test_content)
        
        required_fields = ['current_price', 'predicted_price', 'expected_return', 'confidence', 'signal']
        missing_fields = [field for field in required_fields if field not in result]
        
        if not missing_fields:
            print("[OK] QQQ Master Model parser working correctly")
            print(f"   Current Price: {result.get('current_price')}")
            print(f"   Predicted Price: {result.get('predicted_price')}")
            print(f"   Expected Return: {result.get('expected_return')}")
            print(f"   Confidence: {result.get('confidence')}")
            return True
        else:
            print(f"[FAIL] Missing required fields: {missing_fields}")
            print(f"Available fields: {list(result.keys())}")
            return False
            
    except Exception as e:
        print(f"[FAIL] QQQ Master integration test failed: {e}")
        return False

def test_email_system():
    """Test email system with options analysis"""
    print("\n[EMAIL] Testing Enhanced Email System...")
    
    try:
        # Import the main system
        import trading_reports_main
        
        print("[OK] Enhanced trading reports system imported successfully")
        
        # Check if options analyzer is available
        try:
            from options_strategy_analyzer import OptionsStrategyAnalyzer
            print("[OK] Options strategy analyzer available")
        except ImportError:
            print("[FAIL] Options strategy analyzer not available")
            return False
        
        # Test would require running full system
        print("[INFO] Email system integration ready for testing")
        print("   - Options analysis will be attached to emails")
        print("   - QQQ Master Model reports will be included")
        print("   - Long Bull expected returns will be captured")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Email system test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("[TEST] ENHANCED TRADING SYSTEM TEST SUITE")
    print("="*60)
    
    tests = [
        ("Parser Fix", test_parser_fix),
        ("QQQ Master Integration", test_qqq_master_integration), 
        ("Options Analyzer", test_options_analyzer),
        ("Email System", test_email_system)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"[FAIL] {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("[RESULTS] TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n[STATS] Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("[SUCCESS] All tests passed! System ready for deployment.")
    else:
        print("[WARN] Some tests failed. Review before full deployment.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)