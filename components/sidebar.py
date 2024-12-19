import streamlit as st

def configure_sidebar():
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio("Choose Mode", options=["Content Generator", "Test API"], index=0)

    st.sidebar.markdown("### Temperature Selection")
    temperature = st.sidebar.slider(
        "Select Temperature (Creativity Level)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help=(
            "Low temperature (0.1-0.4) results in focused, deterministic outputs. "
            "High temperature (0.8-1.0) produces more creative but less predictable responses."
        )
    )
    return mode, temperature