"""Mode selector component"""
import streamlit as st

def mode_selector():
    """Render mode selector"""
    return st.selectbox(
        "Select mode:",
        ["normal", "brainstorming", "pro"]
    )






