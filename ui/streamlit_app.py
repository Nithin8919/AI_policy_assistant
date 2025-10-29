"""Streamlit UI"""
import streamlit as st

st.title("AI Policy Assistant")

query = st.text_input("Enter your query:")
mode = st.selectbox("Select mode:", ["normal", "brainstorming", "pro"])

if st.button("Search"):
    st.write("Processing query...")
    # Implementation


