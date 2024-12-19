def display_results(title, script, wiki_research):
    st.subheader("Generated YouTube Title:")
    st.markdown(f"<h3 style='color: #2ECC71;'>{title}</h3>", unsafe_allow_html=True)

    st.subheader("Generated YouTube Script:")
    st.markdown(f"<div style='background-color: #FAFAD2; color: #333; padding: 15px; border-radius: 10px;'>{script}</div>", unsafe_allow_html=True)

    with st.expander("Wikipedia Research"):
        st.write(wiki_research)