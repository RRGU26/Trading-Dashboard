#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test mobile-friendly trading reports email
"""

import sys
import os
sys.path.append('.')

from trading_reports_email import EmailManager, ComprehensiveEmailFormatter
from datetime import datetime

def send_mobile_test_email():
    print("Starting mobile-friendly email test...")

    # Initialize email components
    email_manager = EmailManager()
    email_manager.initialize()
    formatter = ComprehensiveEmailFormatter()

    # Create test data mimicking today's report
    test_reports = {
        'longhorn': {
            'current_price': 577.34,
            'target_price': 572.13,
            'expected_return': -0.9,
            'signal': 'SELL',
            'confidence': 95,
            'risk_level': 'HIGH',
            'report_date': '2025-08-16'
        },
        'qqq_master': {
            'expected_return': -1.13,
            'confidence': 95.0,
            'signal': 'SELL',
            'report_date': '2025-08-16'
        },
        'trading_signal': {
            'current_price': 577.34,
            'signal': 'HOLD',
            'report_date': '2025-08-16'
        },
        'wishing_wealth': {
            'current_price': 577.34,
            'price_prediction': 'QQQ is expected to RISE - Buy TQQQ for leveraged upside',
            'gmi_score': '6/6',
            'recommended_etf': 'TQQQ',
            'suggested_action': 'BUY',
            'signal': 'BUY',
            'confidence': 85,
            'report_date': '2025-08-16'
        },
        'nvidia': {
            'current_price': 180.45,
            'predicted_1_day_price': 178.73,
            'predicted_5_day_price': 171.86,
            'predicted_1_day_return': -0.95,
            'predicted_5_day_return': -4.76,
            'suggested_action': 'SELL',
            'signal': 'SELL',
            'confidence': 85,
            'report_date': '2025-08-16'
        }
    }

    # Generate mobile-friendly HTML
    html_content = formatter.create_comprehensive_html_email(
        model_reports=test_reports,
        dashboard_url='http://localhost:8502',
        db_status_message='All systems operational',
        performance_summary='Test run - Mobile-friendly layout implemented',
        current_date='2025-08-16',
        db_updated=True
    )

    # Send test email
    subject = 'TEST: Mobile-Friendly Trading Report - 2025-08-16'
    recipients = ['RRGU26@gmail.com']

    success = email_manager.send_email(
        sender_email=email_manager.sender_email,
        sender_password=email_manager.sender_password,
        recipient_emails=recipients,
        subject=subject,
        html_body=html_content,
        plain_text_body='Mobile-friendly trading report test. Please check HTML version.'
    )

    if success:
        print('SUCCESS: Mobile-friendly test email sent to RRGU26@gmail.com')
        print('Please check your mobile device to see the improved layout!')
        return True
    else:
        print('FAILED: Could not send test email')
        return False

if __name__ == "__main__":
    send_mobile_test_email()