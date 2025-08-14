# Streamlit Cloud entry point - redirects to main dashboard
import streamlit as st
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import and run the main dashboard
try:
    from dashboard_github import main
    main()
except ImportError as e:
    st.error(f"Failed to import dashboard: {e}")
    st.info("Please ensure all required files are present in the repository.")