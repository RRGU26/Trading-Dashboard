#!/usr/bin/env python3
"""Fix Unicode characters in Long Bull Model"""

import re

# Read the file
with open('QQQ Long Horn Bull Model.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define replacements
replacements = {
    '🚀': '[LAUNCH]',
    '✅': '[OK]',
    '❌': '[ERROR]',
    '💾': '[SAVE]',
    '📊': '[DATA]',
    '📅': '[DATE]',
    '🔥': '[HOT]',
    '📈': '[CHART]',
    '⚡': '[BOLT]',
    '⚠️': '[WARN]',
    '🎯': '[TARGET]',
    '🔴': '[RED]',
    '🟡': '[YELLOW]',
    '🟢': '[GREEN]',
    '🛠️': '[TOOLS]'
}

# Replace all Unicode characters
for unicode_char, replacement in replacements.items():
    content = content.replace(unicode_char, replacement)

# Write the fixed file
with open('QQQ Long Horn Bull Model.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed Unicode characters in QQQ Long Horn Bull Model.py")
print("Replacements made:")
for unicode_char, replacement in replacements.items():
    print(f"  {unicode_char} -> {replacement}")