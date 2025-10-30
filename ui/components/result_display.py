"""Result display component"""
import streamlit as st

def display_results(result):
    """Display query results"""
    st.write("Answer:")
    st.write(result.get("answer", ""))
    
    st.write("Sources:")
    for source in result.get("sources", []):
        st.write(f"- {source}")




